import numpy
import re
from functools import singledispatch
from sympy import latex, Rational, Poly
from sympy.abc import x
from sympy.polys.rings import PolyRing
from sympy.polys.domains import ZZ, QQ, GF
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.domains.field import Field
from sympy.polys.domains.modularinteger import ModularInteger

def parse_matrix_from_string(M, ring):
    """
    Parse a matrix from a string.
    """
    M = M.replace("\n", "").replace("\r", "")
    M = re.sub(r'\s+', ' ', M).strip()
    if M[-1] == ";":
      M = M[:-1]
    
    if "[" in M and "]" in M:
      M = re.sub(r'\s+', '', M)
      M = M.replace("],[", ";").replace("[[", "").replace("]]", "").replace(",", " ")
    
    if ring == ZZ:
      func = int
    elif ring == QQ:
      func = Rational
    elif isinstance(ring, GF):
      func = lambda value: ring(int(value))
    elif isinstance(ring, PolyRing):
      func = lambda value: Poly(value, x, domain=ring.domain)
    
    try:
      return numpy.array([[func(val) for val in row.strip().split(" ")] for row in M.split(";")], dtype=object)
    except Exception as e:
      raise ValueError(f"Error while parsing a matrix {repr(M)} over {repr(ring)}: {e}")

def laTeX(m, leading_coefficient=(0, 0), active_columns=-1):
  
  if isinstance(m, numpy.ndarray):
    entries = m.tolist()
    rows = ["&".join([(laTeX(e) if e != 0 else "\cdot") for e in row]) for row in entries]
    if leading_coefficient[0] and leading_coefficient[0] < m.shape[0]:
      rows[leading_coefficient[0]] = " \\hline " + rows[leading_coefficient[0]]
    
    latex_str = "\\\\".join(rows)
    
    active_columns = active_columns % (1 + m.shape[1])
    r = ["r" for _ in range(m.shape[1])]
    
    if leading_coefficient[1]:
      r[leading_coefficient[1]] = "|r"
    if active_columns and active_columns < m.shape[1]:
      r[active_columns] = "|r"
    
    return "\\left[\\begin{array}" + "{" + "".join(r) + "}" + latex_str + "\\end{array}\\right]"
  elif isinstance(m, Poly):
    if isinstance(m.domain, GF):
      if m == 0:
        return "\\overline{0}"
      else:
        all_coeffs = m.all_coeffs()
        p = m.domain.mod
        degree = m.degree()
        return " + ".join([(f"\\overline{{ {c % p} }}" if c % p != 1 else "") + f"x^{{{(degree-i) if (degree-i) != 1 else ''}}}" for i, c in enumerate(all_coeffs[:-1]) if c != 0]) + ("+" if (degree >= 1) and (all_coeffs[-1] % p) else "") + (f"\\overline{{ {all_coeffs[-1] % p} }}" if all_coeffs[-1] % p else "")
    else:
      return re.sub(r'^.*?left\(|, x,.*$', '', latex(m))
  elif isinstance(m, ModularInteger):
    return f"\\overline{{ {m.val % m.mod} }}"
  elif isinstance(m, numpy.int64):
    return m
  else:
    return latex(m)

def latex_mul_operator(value, value_inv, i, comment=""):
  if isinstance(value, ModularInteger) or (isinstance(value, Poly) and isinstance(value.domain, GF)):
    return comment + f" \( \\mathbf{{Mul}}_{{ \color{{blue}} {i + 1} }}(({laTeX(value)})^{{-1}}={ laTeX(value_inv) }) \)"
  else:
    return comment + f" \( \\mathbf{{Mul}}_{{ \color{{blue}} {i + 1} }}({ laTeX(value_inv) }) \)"

def latex_add_operator(value, neg_value, i, j, comment=""):
  if isinstance(value, ModularInteger) or (isinstance(value, Poly) and isinstance(value.domain, GF)):
    return comment + f" \( \\mathbf{{Add}}_{{ \color{{blue}} {i + 1} \\neq {j + 1} }}(-({laTeX(value)})={ laTeX(neg_value) }) \)"
  else:
    return comment + f" \( \\mathbf{{Add}}_{{ \color{{blue}} {i + 1} \\neq {j + 1} }}({ laTeX(neg_value) }) \)"

def latex_swap_operator(i, j):
  return f"\( \\mathbf{{Swap}}_{{ \color{{blue}} {i + 1} \\neq {j + 1} }} \)"

def echelon_form_column_field(
        M,
        ring,
        leading_coefficient=(0, 0),
        active_columns=-1,
        post_reduction="REF",
        reduction_index=-1,
        show_input=True):
    
    lc_i, lc_j = leading_coefficient
    active_columns = active_columns % (1 + M.shape[1])
    
    if not (lc_i < M.shape[0] and lc_j < active_columns):
      return M, ""
    
    M = M.copy()
    
    html = f"<strong>Current Position ({lc_i + 1}, {lc_j + 1}):</strong>"
    
    if show_input:
      html += "<p style='text-align: center;'>\(M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)<p>"
    
    html += "<ul>"
    
    # Ensure M[0, 0] is either 0 or 1
    if M[lc_i, lc_j] != 0 and M[lc_i, lc_j] != 1:
        html += "<li>" + latex_mul_operator(M[lc_i, lc_j], M[lc_i, lc_j] ** -1, lc_i) + "<br>"
        M[lc_i:lc_i+1] = numpy.array([[ M[lc_i, lc_j] ** -1]]) @ M[lc_i:lc_i+1]
        html += f"\( \leadsto M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)</li>"
    
    close_ul = False
    
    # Loop through the rows of the matrix
    for i in range(lc_i + 1, M.shape[0]):
        if M[i, lc_j] == 0:
            continue

        if M[lc_i, lc_j] == 0:
            html += "<li>" + latex_swap_operator(lc_i, i) + "<br>"
            M[[lc_i, i]] = M[[i, lc_i]]  # Swap rows
            html += f"\( \leadsto M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)</li>"
            
            if M[lc_i, lc_j] != 1:
                html += "<li>" + latex_mul_operator(M[lc_i, lc_j], M[lc_i, lc_j] ** -1, lc_i) + "<br>"
                M[lc_i] = M[lc_i] / M[lc_i, lc_j]
                html += f"\( \leadsto M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)</li>"
        else:
            if close_ul == False:
              close_ul = True
              html += "<li>We perform reduction on the element(s) below the leading coefficient at position " + f"\(({lc_i},{lc_j})\):"
              html += "<ul>"
            
            html += "<li>" + latex_add_operator(M[i, lc_j], -M[i, lc_j], lc_i, i) + "</li>"
            M[i:i+1] -= numpy.array([ [ M[i, lc_j] ] ]) @ M[lc_i:lc_i + 1]
        
    if close_ul:
      html += "</ul>"
      html += "\( \leadsto M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)"
      html += "</li>"
        
    if post_reduction in ["SREF", "RREF"]:
      
      reduction_index = min(lc_i, reduction_index % M.shape[0])
      
      if M[lc_i, lc_j] != 0 and lc_i > 0:
        html += f"<li>We perform the \(\mathbf{{ {post_reduction} }} \) reduction on the element(s) above the leading coefficient at position \(({lc_i + 1}, {lc_j + 1})\):"
        html += "<ul>"
        for i in range(reduction_index):
          if M[i, lc_j] != 0:
            html += "<li>" + latex_add_operator(M[i, lc_j], -M[i, lc_j], lc_i, i) + "</li>"
            M[i:i+1] -= numpy.array([[ M[i, lc_j] ]]) @ M[lc_i:lc_i + 1]
        html += "</ul>"
        html += f"\( \leadsto M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)</li>"
    
    html += f"<li>Column {lc_j + 1} is done!</li>"
    html += "</ul>"
    
    return M, html

@singledispatch
def normalizing_unit(ring : Field, a : object) -> object:
  if a == 0:
    return a, ring.one
  else:
    return a, a ** -1

@normalizing_unit.register
def _(ring : PolyRing, a : object) -> object:
  if a == 0:
    return 0, Poly(1, x, domain=ring.domain)
  else:
    if ring.domain == QQ:
      return a.all_coeffs()[0], Poly(normalizing_unit(ring.domain, ring.domain(str(a.all_coeffs()[0])))[1], x, domain=ring.domain)
    else:
      return a.all_coeffs()[0], Poly(normalizing_unit(ring.domain, ring.domain(int(a.all_coeffs()[0])))[1], x, domain=ring.domain)

@normalizing_unit.register
def _(ring : IntegerRing, a : object) -> object:
  if a == 0:
    return a, ring.one
  else:
    return a, ring(-1) if a < 0 else ring.one

def bezout_matrix(a, b, ring):
    
    if isinstance(ring, PolyRing):
      M = numpy.array([[a, a.one, a.zero], [b, a.zero, a.one]]).astype(type(a))
    else:
      M = numpy.array([[a, ring.one, ring.zero], [b, ring.zero, ring.one]]).astype(object)
    
    html = "<ul>"
    html += f"<li>The input is \(a={laTeX(a)}\) and \(b={laTeX(b)}\):</li>"
    html += "<li>Stack to \(\\begin{bmatrix} a \\\\ b \\end{bmatrix}\) the \(2 \\times 2\) identity matrix \( \leadsto " + laTeX(M, active_columns=1) + "\)</li>"
    
    # if the second element is zero, make sure the first element is positive
    if M[1, 0] == 0:
      val, val_inv = normalizing_unit(ring, M[0, 0])
      if val_inv != 1:
        M[0] *= val_inv
        html += "<li>" + latex_mul_operator(val, val_inv, 0, comment=f"We normalize \(a={laTeX(a)}\);") + "\( \leadsto" + laTeX(M, active_columns=1) + "\) </li>"
    
    # if the second element is not zero, make sure it is positive
    if M[1, 0] != 0:
      val, val_inv = normalizing_unit(ring, M[1, 0])
      if val_inv != 1:
        M[1] *= val_inv
        html += "<li>" + latex_mul_operator(val, val_inv, 1, comment=f"We normalize \(b ={laTeX(b)}\);") + "\( \leadsto" + laTeX(M, active_columns=1) + "\) </li>"
    
    if M[1, 0] != 0:
      html += "<li>\(b \\neq 0\); we perform the reduction until we get a zero:"
      html += "<ul>"
      # apply reduction until the second element is zero
      while M[1, 0] != 0:
          q = divmod(M[0, 0], M[1, 0])
          
          html += "<li><ul>"
          html += f"<li>\( {laTeX(M[0, 0])} = ({laTeX(M[1, 0])}) \cdot ({laTeX(q[0])}) + ({laTeX(q[1])})\)</li>"
          M[0] -= numpy.dot(q[0], M[1])
          html += "<li>" + latex_add_operator(q[0], -q[0], 1, 0) + "\( \leadsto" + laTeX(M, active_columns=1) + "\)</li>"
          
          M[[0, 1]] = M[[1, 0]]
          html += "<li>" + latex_swap_operator(0, 1) + "\( \leadsto" + laTeX(M, active_columns=1) + "\) </li>"
          html += "</ul></li>"
          
      html += "</ul>"
      html += "</li>"
    
    # if the current gcd is not normalized, normalize it
    if M[0, 0] != 0:
      val, val_inv = normalizing_unit(ring, M[0, 0])
      if val_inv != 1:
        M[0] *= val_inv
        html += "<li>" + latex_mul_operator(val, val_inv, 0, comment=f"We normalize the first row;") + "\( \leadsto" + laTeX(M, active_columns=1) + "\) </li>"
    
    html += "<li>Bézout matrix is \(\mathcal{U}\,:=\)" + f"\({laTeX(M[:, 1:])}\)</li>"
    html += f"<li>We can verify \(g:=\mathrm{{gcd}}(a,b)={laTeX(M[0,0])}=({laTeX(M[0,1])})\cdot a+({laTeX(M[0,2])})\cdot b\) and \({laTeX(M[1:,1:])}" + "= \\frac{1}{g}\cdot \\begin{bmatrix} b & -a \\end{bmatrix}\)</li>"
    html += "</ul>"
    
    assert a % M[0, 0] == 0 == b % M[0, 0]
    return M[:, 1:], html

# post_reduction is "REF", "SREF", "RREF"
def echelon_form_column_eukledian_ring(M, ring,
        leading_coefficient=(0, 0),
        active_columns=-1,
        post_reduction="REF",
        reduction_index=-1,
        show_input=True):
    
    lc_i, lc_j = leading_coefficient
    
    active_columns = active_columns % (1 + M.shape[1])
    
    if not (lc_i < M.shape[0] and lc_j < active_columns):
      return M, ""
    
    M = M.copy()
    
    html = f"<strong>Current Position ({lc_i + 1}, {lc_j + 1}):</strong>"
    
    if show_input:
      html += "<p style='text-align: center;'>\(M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)<p>"
    
    html += "<ul>"
    
    # Loop through the rows starting from the second row
    for i in range(lc_i + 1, M.shape[0]):
        if M[i, lc_j] == 0:
            continue
        html += f"<li>We consider the position ({i+1}, {lc_j + 1})"
        html += "<ul>"
        html += "<li>We compute the Bézout matrix \(\mathcal{U}_{" + f"{lc_i + 1} ≠ {i + 1} " + "}" + f"({laTeX(M[lc_i, lc_j])}, {laTeX(M[i, lc_j])})\):</li>"
        bezout_mat, html_ = bezout_matrix(M[lc_i, lc_j], M[i, lc_j], ring)
        
        html += "<li>" + html_ + "</li>"
        html += f"<li>We transform the rows \({lc_i + 1}\) and \({i + 1}\) of \( M \) using " + "\(\mathcal{U}_{" + f"{lc_i + 1} ≠ {i + 1}" + "}" + f"({laTeX(M[lc_i, lc_j])}, {laTeX(M[i, lc_j])})\):"
        M[[lc_i, i]] = bezout_mat @ M[[lc_i, i]]
        html += "<p style='text-align: left;'> \( \leadsto M=" + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)" + "</p></li>"
        html += "</ul>"
        html += "</li>"
    
    if M[lc_i, lc_j] != 0:
        val, val_inv = normalizing_unit(ring, M[lc_i, lc_j])
        if val_inv != 1:
          html += f"<li>We consider row {lc_i + 1}</li>"
          html += "<li>" + latex_mul_operator(val, val_inv, lc_i)
          M[lc_i] *= val_inv
          html += "<p style='text-align: left;'> \(M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)</p></li>"
    
    if post_reduction == "SREF":
      
      reduction_index = min(lc_i, reduction_index % M.shape[0])
      
      if M[lc_i, lc_j] != 0 and reduction_index > 0:
        html += "<li>We perform \(\mathbf{SREF}\) reduction on the " + f"{reduction_index} element(s) above the leading coefficient position \(({lc_i + 1}, {lc_j + 1})\):"
        html += "<ul>"
        for i in range(reduction_index):
            if M[i, lc_j] != 0:
              q = divmod(M[i, lc_j], M[lc_i, lc_j])
              if q[1] == 0:
                m = f"\({laTeX(M[i, lc_j])} = {laTeX(M[lc_i, lc_j])}\cdot({laTeX(q[0])})\);"
                html += "<li>" + latex_add_operator(q[0], -q[0], lc_i, i, comment=m) + "</li>"
                M[i] -= numpy.dot(q[0], M[lc_i])
        html += "</ul>"
        html += "<p style='text-align: left;'> \(\leadsto M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)" + "</p>"
        html += "</li>"
        
    elif post_reduction == "RREF":
      
      reduction_index = min(lc_i, reduction_index % M.shape[0])
      
      if M[lc_i, lc_j] != 0 and reduction_index > 0:
        html += "<li>We perform \(\mathbf{RREF}\) reduction on the " + f"{reduction_index} element(s) above the leading coefficient position \(({lc_i + 1}, {lc_j + 1})\):"
        html += "<ul>"
        for i in range(reduction_index):
            if M[i, lc_j] != 0:
              q = divmod(M[i, lc_j], M[lc_i, lc_j])
              m = f"\({laTeX(M[i, lc_j])} = {laTeX(M[lc_i, lc_j])}\cdot({laTeX(q[0])}) + {laTeX(q[1])}\);"
              html += "<li>" + latex_add_operator(q[0], -q[0], lc_i, i, comment=m) + "</li>"
              M[i] -= numpy.dot(q[0], M[lc_i])
        html += "</ul>"
        html += "<p style='text-align: left;'> \(\leadsto M = " + laTeX(M, leading_coefficient=leading_coefficient, active_columns=active_columns) + "\)" + "</p>"
        html += "</li>"
    
    html += f"<li>Column {lc_j + 1} is done!</li>"
    html += "</ul>"
    
    return M, html
    
def echelon_form_matrix(M, ring,
        leading_coefficient=(0, 0),
        active_columns=-1,
        post_reduction="REF",
        reduction_index=-1):
    
    if ring == ZZ or isinstance(ring, PolyRing):
      echelon_form_column = echelon_form_column_eukledian_ring
    else:
      echelon_form_column = echelon_form_column_field
    
    if M.shape[0] == 0 or M.shape[1] == 0:
        return M, ""
    
    lc_i, lc_j = leading_coefficient
    
    M, html = echelon_form_column(M, ring,
                        leading_coefficient=leading_coefficient,
                        active_columns=active_columns,
                        post_reduction=post_reduction,
                        reduction_index=reduction_index)
    
    while lc_j < M.shape[1] - 1 and lc_i < M.shape[0]:
      
      if M[lc_i, lc_j] != 0:
        lc_i += 1
      
      lc_j += 1
      
      M, html_ = echelon_form_column(M, ring,
                            leading_coefficient=(lc_i, lc_j),
                            active_columns=active_columns,
                            post_reduction=post_reduction,
                            reduction_index=reduction_index,
                            show_input=True)
      
      html += "<hr>" + html_
      
    return M, html
