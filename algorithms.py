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

def convert_matrix_to_string(M, ring):
  if isinstance(ring, PolyRing):
      return str([[poly.expr for poly in row] for row in M.tolist()])
  else:
      return str(M.tolist())
  
def laTeX(m, leading_coefficient=(0, 0), active_columns=-1):
  
  if isinstance(m, numpy.ndarray):
    
    if m.shape[0] == 0 or m.shape[1] == 0:
      return f"\\mathbf{0}_{{ {m.shape[0]} \\times {m.shape[1]} }}"
    
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

def zero_matrix(m, n, ring, dtype):
    
    if ring in [ZZ, QQ]:
      zero = 0
    elif isinstance(ring, GF):
      zero = ring(0)
    elif isinstance(ring, PolyRing):
      zero = Poly("0", x, domain=ring.domain)
    
    return numpy.full((m, n), zero, dtype=dtype)

def eye_matrix(n, ring, dtype):
    
    if ring in [ZZ, QQ]:
      one = 1
    elif isinstance(ring, GF):
      one = ring(1)
    elif isinstance(ring, PolyRing):
      one = Poly("1", x, domain=ring.domain)
    
    mat = zero_matrix(n, n, ring, dtype)
    
    for i in range(n):
      mat[i,i] = one
    
    return mat

def solve_left_linear_system(A, B, ring, post_reduction="REF", active_columns=None):
    
    nr_variables = A.shape[0]
    nr_equations = A.shape[1]
    nr_systems = B.shape[0]
    
    variables = [f"X_{ {i+1} }" for i in range(nr_variables)]
    
    if nr_systems == 1 and ring in [ZZ, QQ]:
      x_latex = "\\begin{bmatrix}" + " & ".join(variables) + "\\end{bmatrix}"
      system_latex = "\\begin{array}{r}" + "\\\\".join([a+b for a, b in zip([" & ".join([f"{'+' if val>=0 else '-'} {abs(val)}{var}" for val, var in zip(A[:,i], variables)]) for i in range(nr_equations)], [f"& = & {val}" for val in B[0,:]])]) + "\\end{array}"
    elif nr_systems == 1 and ring not in [ZZ, QQ]:
      x_latex = "\\begin{bmatrix}" + " & ".join(variables) + "\\end{bmatrix}"
      system_latex = "\\begin{array}{r}" + "\\\\".join([a+b for a, b in zip([" & ".join([f"+ ({laTeX(val)}){var}" for val, var in zip(A[:,i], variables)]) for i in range(nr_equations)], [f"& = & ({laTeX(val)})" for val in B[0,:]])]) + "\\end{array}"
    else:
      x_latex = "\\begin{bmatrix}" + "\\\\".join([" & ".join(["X_{" +f"{j+1},{i+1}" + "}" for i in range(nr_variables)]) for j in range(nr_systems)]) + "\\end{bmatrix}"
      system_latex = "X \cdot A = B"
    
    if ring in [ZZ, QQ]:
      finite_ring = False
      ring_latex = " \mathbb{Z}" if ring == ZZ else " \mathbb{Q}"
    elif isinstance(ring, GF):
      finite_ring = True
      ring_latex = f"\\mathbb{{F}}_{{ {ring.mod} }}"
    elif isinstance(ring, PolyRing):
      finite_ring = False
      if ring.domain == QQ:
        ring_latex = " \mathbb{Q}[x]"
      elif isinstance(ring.domain, GF):
        ring_latex = f"\\mathbb{{F}}_{{ {ring.domain.mod} }}[x]"
    
    id_mat = eye_matrix(nr_variables, ring, A.dtype)
    
    M = numpy.hstack([A, id_mat])
    
    xREF_M = echelon_form_matrix(M,
                  leading_coefficient=(0, 0),
                  post_reduction=post_reduction,
                  active_columns=nr_equations if active_columns is None else active_columns,
                  ring=ring
              )[0]
    
    A_tilde = xREF_M[:, :nr_equations]
    I_tilde = xREF_M[:, nr_equations:]
    
    rank_A = len([i for i in range(nr_variables) if numpy.any(A_tilde[i,:] != 0)])
    
    G = A_tilde[:rank_A, :]
    Z = I_tilde[:rank_A, :]
    S = I_tilde[rank_A:, :]
    
    nr_syzygies = nr_variables - rank_A
    S_latex = laTeX(S)
    
    if nr_systems == 1:
      parametrized_solution_latex = "\\begin{bmatrix}" + " & ".join([f"t_{ {i+1} }" for i in range(nr_syzygies)]) + "\\end{bmatrix} \cdot" + S_latex + "|" + ",".join([f"t_{ {i+1} }" for i in range(nr_syzygies)]) + "\in" + ring_latex
    else:
      parametrized_solution_latex = "\\begin{bmatrix}" + "\\\\".join([" & ".join(["t_{" +f"{j+1},{i+1}" + "}" for i in range(nr_syzygies)]) for j in range(nr_systems)]) + "\\end{bmatrix} \cdot" + S_latex + "| t_{i,j} \in" + ring_latex + "\mbox{ for }" + f"i \leq {nr_systems}, j \leq {nr_syzygies}"
    
    zero_mat = zero_matrix(nr_systems, nr_variables, ring, A.dtype)
    
    H = numpy.block([[-B, zero_mat], [G, Z]])
    
    SREF_H = echelon_form_matrix(H,
                leading_coefficient=(nr_systems, 0),
                post_reduction="SREF",
                reduction_index=nr_systems,
                active_columns=nr_equations,
                ring=ring
              )[0]
    
    O = -SREF_H[:nr_systems, :nr_equations]
    X_p = SREF_H[:nr_systems, nr_equations:]
    
    is_solvable = numpy.all(O == 0)
    
    if is_solvable:
      if nr_syzygies == 0:
        solution_set_latex = "\\left\\{" + laTeX(X_p) + "\\right\\}"
      else:
        solution_set_latex = "\\left\\{" + laTeX(X_p) + " + " + parametrized_solution_latex + "\\right\\}"
    else:
      solution_set_latex = "\\emptyset"
    
    solution_set_latex = solution_set_latex + "\\subseteq" + ring_latex + "^{" + str(nr_systems) + "\\times" + str(nr_variables) + "}"
    
    solution_set_text = convert_matrix_to_string(numpy.vstack([X_p, S]), ring)
    
    return X_p, S, dict(ring=ring,
        finite_ring=finite_ring,
        ring_latex=ring_latex,
        system_latex=system_latex,
        A_latex=laTeX(A),
        B_latex=laTeX(B),
        x_latex=x_latex,
        M_latex=laTeX(M, active_columns=nr_equations),
        xREF_M_latex=laTeX(xREF_M, active_columns=nr_equations),
        post_reduction=post_reduction,
        rank_A=rank_A,
        G_latex=laTeX(G),
        Z_latex=laTeX(Z),
        S_latex=laTeX(S),
        nr_syzygies=nr_syzygies,
        nr_variables=nr_variables,
        nr_equations=nr_equations,
        nr_systems=nr_systems,
        parametrized_solution_latex=parametrized_solution_latex,
        zero_solution_latex=laTeX(numpy.zeros((nr_systems, nr_variables), dtype=A.dtype)),
        H_latex=laTeX(H, leading_coefficient=(nr_systems, 0), active_columns=nr_equations),
        SREF_H_latex=laTeX(SREF_H, leading_coefficient=(nr_systems, 0), active_columns=nr_equations),
        O_latex=laTeX(O),
        X_p_latex=laTeX(X_p),
        is_solvable=is_solvable,
        solution_set_latex=solution_set_latex,
        solution_set_text=solution_set_text)

def solve_right_linear_system(A, B, ring, post_reduction="REF", active_rows=None):
  
  X_p, S, info = solve_left_linear_system(A.T, B.T, ring, post_reduction=post_reduction, active_columns=active_rows)
  
  if info["is_solvable"]:
    parametrized_solution_latex = ""
    if info['nr_systems'] == 1:
      if info["nr_syzygies"] != 0:
        parametrized_solution_latex = " + " + laTeX(S.T)
        parametrized_solution_latex += "\cdot"
        parametrized_solution_latex += "\\begin{bmatrix}" + " \\\\ ".join([f"t_{ {i+1} }" for i in range(info["nr_syzygies"])]) + "\\end{bmatrix}"
        parametrized_solution_latex += "|" + ",".join([f"t_{ {i+1} }" for i in range(info["nr_syzygies"])]) + "\in" + info["ring_latex"]
    else:
      if info["nr_syzygies"] != 0:
        parametrized_solution_latex = " + " + laTeX(S.T)
        parametrized_solution_latex += "\cdot"
        parametrized_solution_latex += "\\begin{bmatrix}" + "\\\\".join([" & ".join(["t_{" + f"{j+1},{i+1}" + "}" for i in range(info["nr_systems"])]) for j in range(info["nr_syzygies"])]) + "\\end{bmatrix}"
        parametrized_solution_latex += "| t_{i,j} \in" + info["ring_latex"] + "\mbox{ for }" + f"i \leq {info['nr_syzygies']}, j \leq {info['nr_systems']}"
    
    solution_set_latex = "\\left\\{" + laTeX(X_p.T) + parametrized_solution_latex + "\\right\\}"
    solution_set_latex += "\\subseteq" + info["ring_latex"] + "^{" + str(info["nr_variables"]) + "\\times" + str(info["nr_systems"]) + "}"
    return X_p.T, S.T, solution_set_latex
  else:
    return None, None, info["solution_set_latex"]

def solve_left_linear_system_gauss_elimination(A, B, ring):
  
  nr_variables = A.shape[0]
  nr_equations = A.shape[1]
  nr_systems = B.shape[0]
  
  if ring == QQ:
    one = 1
    finite_ring = False
    ring_latex = " \mathbb{Z}" if ring == ZZ else " \mathbb{Q}"
  elif isinstance(ring, GF):
    one = ring(1)
    finite_ring = True
    ring_latex = f"\\mathbb{{F}}_{{ {ring.mod} }}"
  
  variables = [f"X_{ {i+1} }" for i in range(nr_variables)]
  
  if nr_systems == 1 and ring == QQ:
    x_latex = "\\begin{bmatrix}" + " & ".join(variables) + "\\end{bmatrix}"
    system_latex = "\\begin{array}{r}" + "\\\\".join([a+b for a, b in zip([" & ".join([f"{'+' if val>=0 else '-'} {abs(val)}{var}" for val, var in zip(A[:,i], variables)]) for i in range(nr_equations)], [f"& = & {val}" for val in B[0,:]])]) + "\\end{array}"
  elif nr_systems == 1 and ring != QQ:
    x_latex = "\\begin{bmatrix}" + " & ".join(variables) + "\\end{bmatrix}"
    system_latex = "\\begin{array}{r}" + "\\\\".join([a+b for a, b in zip([" & ".join([f"+ ({laTeX(val)}){var}" for val, var in zip(A[:,i], variables)]) for i in range(nr_equations)], [f"& = & ({laTeX(val)})" for val in B[0,:]])]) + "\\end{array}"
  else:
    x_latex = "\\begin{bmatrix}" + "\\\\".join([" & ".join(["X_{" +f"{j+1},{i+1}" + "}" for i in range(nr_variables)]) for j in range(nr_systems)]) + "\\end{bmatrix}"
    system_latex = "X \cdot A = B"
  
  M = numpy.vstack([A, B])
  
  RREF_M = (echelon_form_matrix(M.T,
                leading_coefficient=(0, 0),
                post_reduction="RREF",
                active_columns=-1,
                ring=ring
              )[0]).T
  
  indices_A = [i for i in range(nr_equations) if numpy.any(RREF_M[:nr_variables, i] != 0)]
  rank_A = len(indices_A)
  
  indices_M = [i for i in range(nr_equations) if numpy.any(RREF_M[:, i] != 0)]
  rank_M = len(indices_M)
  
  is_solvable = rank_A == rank_M
  
  pivot_rows_A = set(next(j for j in range(nr_variables) if RREF_M[j,i] != 0) for i in range(rank_A))
  non_pivot_rows_A = set(j for j in range(nr_variables) if j not in pivot_rows_A)
  
  eRREF_M = RREF_M[:,:rank_A].copy()
  
  if is_solvable:
    for i in range(nr_variables):
      if i not in pivot_rows_A:
        eRREF_M = numpy.insert(eRREF_M, i, numpy.zeros(nr_variables + nr_systems, dtype=A.dtype), axis=1)
        eRREF_M[i,i] = -one
    
    X_p = eRREF_M[nr_variables:, :]
    
    S = eRREF_M[list(non_pivot_rows_A),:]
    
    nr_syzygies = nr_variables - rank_A
    
    if nr_syzygies > 0:
      if nr_systems == 1:
        parametrized_solution_latex = "\\begin{bmatrix}" + " & ".join([f"t_{ {i+1} }" for i in range(nr_syzygies)]) + "\\end{bmatrix} \cdot" + laTeX(S) + "|" + ",".join([f"t_{ {i+1} }" for i in range(nr_syzygies)]) + "\in" + ring_latex
      else:
        parametrized_solution_latex = "\\begin{bmatrix}" + "\\\\".join([" & ".join(["t_{" +f"{j+1},{i+1}" + "}" for i in range(nr_syzygies)]) for j in range(nr_systems)]) + "\\end{bmatrix} \cdot" + laTeX(S) + "| t_{i,j} \in" + ring_latex + "\mbox{ for }" + f"i \leq {nr_systems}, j \leq {nr_syzygies}"
    else:
      parametrized_solution_latex = f"\\mathbf{{0}}_{{ {nr_systems} \\times {nr_variables} }}"
    
    solution_set_text = convert_matrix_to_string(numpy.vstack([X_p, S]), ring)
    solution_set_latex = "\\left\\{" + laTeX(X_p) + (("+" + parametrized_solution_latex) if nr_syzygies else "") + "\\right\\}"
    solution_set_latex += "\\subseteq" + ring_latex + "^{" + f"{nr_systems}" + "\\times" + f"{nr_variables}" + "}"
  
  else:
    X_p = None
    S = None
    nr_syzygies = None
    solution_set_text = "[]"
    solution_set_latex = "\\emptyset"
    parametrized_solution_latex = f"\\mathbf{{0}}_{{ {nr_systems} \\times {nr_variables} }}"
    
  return X_p, S, dict(
    x_latex=x_latex,
    system_latex=system_latex,
    finite_ring=finite_ring,
    ring=ring,
    ring_latex=ring_latex,
    A_latex=laTeX(A),
    B_latex=laTeX(B),
    M_latex=laTeX(M, leading_coefficient=(nr_variables, 0)),
    S_latex=laTeX(S),
    X_p_latex=laTeX(X_p),
    solution_set_latex=solution_set_latex,
    solution_set_text=solution_set_text,
    pivot_rows_A={i + 1 for i in pivot_rows_A} if len(pivot_rows_A) > 0 else "",
    non_pivot_rows_A={i + 1 for i in non_pivot_rows_A} if len(non_pivot_rows_A) > 0 else "",
    post_reduction="RREF",
    RREF_M_latex=laTeX(RREF_M, leading_coefficient=(nr_variables, 0)),
    RREF_M_latex_=laTeX(RREF_M[:,:rank_M], leading_coefficient=(nr_variables, 0)),
    eRREF_M_latex=laTeX(eRREF_M, leading_coefficient=(nr_variables, 0)),
    minus_one_latex=laTeX(-one),
    rank_A=rank_A,
    rank_M=rank_M,
    is_solvable=is_solvable,
    parametrized_solution_latex=parametrized_solution_latex,
    nr_variables=nr_variables,
    nr_systems=nr_systems,
    nr_syzygies=nr_syzygies,
    indices_A=indices_A,
    indices_M=indices_M
  )
  
  
  
  
def left_inverses(A, ring):
  
    B = eye_matrix(A.shape[1], ring, A.dtype)
    
    X_p, S, info = solve_left_linear_system(A, B, ring, post_reduction="RREF", active_columns=-1)
    
    if info["is_solvable"]:
      return X_p, S, info["solution_set_latex"]
    else:
      return None, None, info["solution_set_latex"]

def right_inverses(A, ring):
    
    B = eye_matrix(A.shape[0], ring, A.dtype)
    
    return solve_right_linear_system(A, B, ring, post_reduction="RREF", active_rows=-1)
  