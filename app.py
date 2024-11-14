import numpy
import sympy
from flask import (Flask, redirect, render_template, request, url_for)
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, StringField, IntegerField, SelectField
from wtforms.validators import DataRequired, ValidationError
from math import *

app = Flask(__name__)
app.config["SECRET_KEY"] = "2934d_AaSsaIXxg_S_!_SDXjXD!US??Ksm!--S;SUI§0KMSK"

def characteristic_validator(form, field):
    if not sympy.isprime(field.data):
        raise ValidationError("The characteristic must be a prime number.")

class MatrixForm(FlaskForm):
    M = TextAreaField("\(M=~\)", validators=[DataRequired()])
    RING = SelectField(
        'The matrix is defined over:',
        choices=[ ('Rationals', 'Rationals'),
                  ('Integers', 'Integers'),
                  ('FiniteField', 'Finite field'),
                  ('PolynomialRing', 'Polynomial ring'),
                ],
        default='Rationals')
    DOMAIN = SelectField(
        'The polynomial ring \(k[x]\) for \(k=\)',
        choices=[ ('Rationals', 'Rationals'),
              ('FiniteField', 'Finite field'),
              ],
        default='Rationals')
    CHAR = IntegerField(
        "The finite field \(\mathbb{F}_{p}\) for the prime number \(p=\)",
        validators=[DataRequired(), characteristic_validator],
        default=2)
    LEADING_COEFFICIENT_POSITION = StringField("Starting leading coefficient position:", default='1, 1')
    ACTIVE_COLUMNS = IntegerField("Apply algorithm only to the first \(c\) columns: \(c\) = ", default=-1)
    ART_OF_REDUCTION = SelectField(
        'Reduction type:',
        choices=[('REF', 'Row Echelon Form (REF)'),
                 ('SREF', 'Semi-Reduced Row Echelon Form (SREF)'),
                 ('RREF', 'Reduced Row Echelon Form (RREF)'),],
        default='REF')
    REDUCE_THE_FIRST_N_ROWS = IntegerField("Use each leading coefficient to reduce elements only in the first \(r\) rows above it: \(r\) = ", default=-1)
    submit = SubmitField("Submit")

class LinearSystemForm(FlaskForm):
    
    A = TextAreaField("\(A=~\)", validators=[DataRequired()])
    B = TextAreaField("\(B=~\)", validators=[DataRequired()])
    RING = SelectField(
        'The linear system is defined over:',
        choices=[ ('Rationals', 'Rationals'),
              ('Integers', 'Integers'),
              ('FiniteField', 'Finite field'),
              ('PolynomialRing', 'Polynomial ring'),
              ],
        default='Rationals')
    DOMAIN = SelectField(
        'The polynomial ring \(k[x]\) for \(k=\)',
        choices=[ ('Rationals', 'Rationals'),
              ('FiniteField', 'Finite field'),
              ],
        default='Rationals')
    CHAR = IntegerField(
        "The finite field \(\mathbb{F}_{p}\) for the prime number \(p=\)",
        validators=[DataRequired()],
        default=2)
    submit = SubmitField("Solve System")
    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/row-echelon-form', methods=['GET', 'POST'])
def row_echelon_form():
    form = MatrixForm()
    if form.validate_on_submit():
        leading_coefficient = form.LEADING_COEFFICIENT_POSITION.data
        leading_coefficient = tuple([int(i) - 1 for i in leading_coefficient.replace("(","").replace(")", "").replace(" ", "").split(",")])
        if any(i < 0 for i in leading_coefficient):
          raise ValueError("The starting leading coefficient position must consist of two positive integers")
        
        active_columns = form.ACTIVE_COLUMNS.data
        post_reduction = form.ART_OF_REDUCTION.data
        
        ring_data = form.RING.data
        
        char = form.CHAR.data
        
        if ring_data == "Integers":
          ring = ZZ
          ring_latex = "\mathbb{Z}"
          is_field = False
        elif ring_data == "Rationals":
          ring = QQ
          ring_latex = "\mathbb{Q}"
          is_field = True
        elif ring_data == "FiniteField":
          ring = GF(char)
          ring_latex = f"\mathbb{{F}}_{{ {char} }}"
          is_field = True
        elif ring_data == "PolynomialRing":
          domain_data = form.DOMAIN.data
          if domain_data == "Rationals":
            ring, _ = sympy.ring("x", domain=QQ)
            ring_latex = "\mathbb{Q}[x]"
          elif domain_data == "FiniteField":
            ring, _ = sympy.ring("x", domain=GF(char))
            ring_latex = f"\mathbb{{F}}_{{ {char} }}[x]"
          is_field = False
        else:
          raise ValueError("The ring must be either 'Integers', 'Rationals', 'FiniteField' or 'PolynomialRing'")
        
        M = parse_matrix_from_string(form.M.data, ring)
        
        reduction_index = form.REDUCE_THE_FIRST_N_ROWS.data
        reduction_index = min(reduction_index, M.shape[0] - 1)
        
        output = echelon_form_matrix(M, ring,
                      leading_coefficient=leading_coefficient,
                      active_columns=active_columns,
                      post_reduction=post_reduction,
                      reduction_index=reduction_index)
        
        if isinstance(ring, PolyRing):
          output_as_text = str([[poly.expr for poly in row] for row in output[0].tolist()])
        else:
          output_as_text = str(output[0].tolist())
        
        rank_M = len([i for i in range(M.shape[0]) if numpy.any(output[0][i,:] != 0)])
        
        left_invertible = "unknown"
        
        if rank_M == M.shape[1]:
          b = numpy.array_equal(output[0][:rank_M, :], numpy.eye(rank_M, dtype=M.dtype))
          
          if b:
            left_invertible = True
          
          if post_reduction in ["SREF", "RREF"]:
            left_invertible = b
          
        output =  laTeX(output[0], leading_coefficient=leading_coefficient, active_columns=active_columns), output[1]
        
        return render_template('row-echelon-form.html',
                    form=form,
                    ring_latex=ring_latex,
                    is_field=is_field,
                    left_invertible=left_invertible,
                    post_reduction=post_reduction,
                    output=output,
                    output_as_text = output_as_text,
                    rank_M=rank_M,
                    nr_rows_M=M.shape[0],
                    nr_cols_M=M.shape[1])
    else:
        return render_template('row-echelon-form.html', form=form)

@app.route('/solve-linear-system', methods=['GET', 'POST'])
def solve_linear_system():
  
  form = LinearSystemForm()
  
  if request.method == 'GET':
    form.A.data = request.args.get('A', "")
    form.B.data = request.args.get('B', "")
  
  if form.validate_on_submit():
    
    ring_data = form.RING.data
    
    char = form.CHAR.data if sympy.isprime(form.CHAR.data) else sympy.nextprime(form.CHAR.data)
    
    if ring_data == "Integers":
      ring = ZZ
      finite_ring = False
      ring_latex = "\mathbb{Z}"
    elif ring_data == "Rationals":
      ring = QQ
      finite_ring = False
      ring_latex = "\mathbb{Q}"
    elif ring_data == "FiniteField":
      ring = GF(char)
      finite_ring = True
      ring_latex = f"\mathbb{{F}}_{{ {ring.mod} }}"
    elif ring_data == "PolynomialRing":
      domain_data = form.DOMAIN.data
      if domain_data == "Rationals":
        ring, _ = sympy.ring("x", domain=QQ)
        ring_latex = "\mathbb{Q}[x]"
        finite_ring = False
      elif domain_data == "FiniteField":
        ring, _ = sympy.ring("x", domain=GF(char))
        ring_latex = f"\mathbb{{F}}_{{ {char} }}[x]"
        finite_ring = False
    else:
      raise ValueError("The ring must be either 'Integers', 'Rationals', 'FiniteField' or 'PolynomialRing'")
    
    A = parse_matrix_from_string(form.A.data, ring)
    B = parse_matrix_from_string(form.B.data, ring)
    
    error = False
    
    if A.shape[1] != B.shape[1]:
      error="The number of columns in \(A\) and \(B\) must be the same."
    
    if error:
      return render_template('solve-linear-system.html',
              form=form,
              output=False,
              error=error)
    
    nr_variables = A.shape[0]
    nr_equations = A.shape[1]
    nr_systems = B.shape[0]
    
    if nr_systems == 1 and ring in ["Integers", "Rationals"]:
      variables = [f"X_{ {i+1} }" for i in range(nr_variables)]
      x_latex = "\\begin{bmatrix}" + " & ".join(variables) + "\\end{bmatrix}"
      system_latex = "\\begin{array}{r}" + "\\\\".join([a+b for a, b in zip([" & ".join([f"{'+' if val>=0 else '-'} {abs(val)}{var}" for val, var in zip(A[:,i], variables)]) for i in range(nr_equations)], [f"& = & {val}" for val in B[0,:]])]) + "\\end{array}"
    elif nr_systems == 1 and ring not in ["Integers", "Rationals"]:
      variables = [f"X_{ {i+1} }" for i in range(nr_variables)]
      x_latex = "\\begin{bmatrix}" + " & ".join(variables) + "\\end{bmatrix}"
      system_latex = "\\begin{array}{r}" + "\\\\".join([a+b for a, b in zip([" & ".join([f"+ ({laTeX(val)}){var}" for val, var in zip(A[:,i], variables)]) for i in range(nr_equations)], [f"& = & ({laTeX(val)})" for val in B[0,:]])]) + "\\end{array}"
    else:
      x_latex = "\\begin{bmatrix}" + "\\\\".join([" & ".join(["X_{" +f"{j+1},{i+1}" + "}" for i in range(nr_variables)]) for j in range(nr_systems)]) + "\\end{bmatrix}"
      system_latex = "X \cdot A = B"
    
    A_latex = laTeX(A)
    B_latex = laTeX(B)
    
    if ring in [ZZ, QQ]:
      zero = 0
      one = 1
    elif isinstance(ring, GF):
      zero = ring(0)
      one = ring(1)
    elif isinstance(ring, PolyRing):
      zero = Poly("0", x, domain=ring.domain)
      one = Poly("1", x, domain=ring.domain)
    
    id_mat = numpy.full((nr_variables, nr_variables), zero, dtype=A.dtype)
    for i in range(nr_variables):
      id_mat[i,i] = one
    
    M = numpy.hstack([A, id_mat])
    M_latex = laTeX(M, active_columns=nr_equations)
    
    
    REF_M = echelon_form_matrix(M,
                  leading_coefficient=(0, 0),
                  post_reduction="REF",
                  active_columns=nr_equations,
                  ring=ring
              )[0]
    
    REF_M_latex = laTeX(REF_M, active_columns=nr_equations)
    
    A_tilde = REF_M[:, :nr_equations]
    I_tilde = REF_M[:, nr_equations:]
    
    rank_A = len([i for i in range(nr_variables) if numpy.any(A_tilde[i,:] != 0)])
    
    G = A_tilde[:rank_A, :]
    G_latex = laTeX(G)
    
    Z = I_tilde[:rank_A, :]
    Z_latex = laTeX(Z)
    
    S = I_tilde[rank_A:, :]
    S_latex = laTeX(S)
    
    nr_syzygies = nr_variables - rank_A
    
    if nr_systems == 1:
      parametrized_solution_latex = "\\begin{bmatrix}" + " & ".join([f"t_{ {i+1} }" for i in range(nr_syzygies)]) + "\\end{bmatrix} \cdot" + S_latex + "|" + ",".join([f"t_{ {i+1} }" for i in range(nr_syzygies)]) + "\in" + ring_latex
    else:
      parametrized_solution_latex = "\\begin{bmatrix}" + "\\\\".join([" & ".join(["t_{" +f"{j+1},{i+1}" + "}" for i in range(nr_syzygies)]) for j in range(nr_systems)]) + "\\end{bmatrix} \cdot" + S_latex + "| t_{i,j} \in" + ring_latex + "\mbox{ for }" + f"i \leq {nr_systems}, j \leq {nr_syzygies}"
    
    zero_solution_latex = laTeX(numpy.zeros((nr_systems, nr_variables), dtype=A.dtype))
    
    id_1 = numpy.full((nr_systems, nr_systems), zero, dtype=A.dtype)
    for i in range(nr_systems):
      id_1[i,i] = one
    
    zero_mat = numpy.full((nr_systems, nr_variables), zero, dtype=A.dtype)
    
    H = numpy.block([[-B, zero_mat], [G, Z]])
    
    H_latex = laTeX(H, leading_coefficient=(nr_systems, 0), active_columns=nr_equations)
    
    SREF_H = echelon_form_matrix(H,
                leading_coefficient=(nr_systems, 0),
                post_reduction="SREF",
                reduction_index=nr_systems,
                active_columns=nr_equations,
                ring=ring
              )[0]
    
    SREF_H_latex = laTeX(SREF_H, leading_coefficient=(nr_systems, 0), active_columns=nr_equations)
    
    O = -SREF_H[:nr_systems, :nr_equations]
    O_latex = laTeX(O)
    
    X_p = SREF_H[:nr_systems, nr_equations:]
    X_p_latex = laTeX(X_p)
    
    if numpy.all(O == 0):
      is_solvable = True
    else:
      is_solvable = False
    
    return render_template('solve-linear-system.html',
              output=True,
              form=form,
              ring=ring,
              finite_ring=finite_ring,
              ring_latex=ring_latex,
              system_latex=system_latex,
              A_latex=A_latex,
              B_latex=B_latex,
              x_latex=x_latex,
              M_latex=M_latex,
              REF_M_latex=REF_M_latex,
              rank_A=rank_A,
              G_latex=G_latex,
              Z_latex=Z_latex,
              S_latex=S_latex,
              nr_syzygies=nr_syzygies,
              nr_variables=nr_variables,
              nr_equations=nr_equations,
              nr_systems=nr_systems,
              parametrized_solution_latex=parametrized_solution_latex,
              zero_solution_latex=zero_solution_latex,
              H_latex=H_latex,
              SREF_H_latex=SREF_H_latex,
              O_latex=O_latex,
              X_p_latex=X_p_latex,
              is_solvable=is_solvable
            )
  
  return render_template('solve-linear-system.html',
              form=form,
              output=False,
            )
  
@app.route('/create-linear-system', methods=['GET', 'POST'])
def create_linear_system():
    nr_equations = int(request.args.get('nr_equations', 1))
    nr_variables = int(request.args.get('nr_variables', 1))
    return render_template('create-linear-system.html', nr_equations=nr_equations, nr_variables=nr_variables)

@app.route('/process_matrix/<int:nr_equations>/<int:nr_variables>', methods=['POST'])
def process_matrix(nr_equations, nr_variables):
    
    A = ""
    for j in range(nr_variables):
        row = ""
        for i in range(nr_equations):
            cell_value = request.form.get(f'matrix_{i}_{j}')
            row += cell_value + " " if i < nr_equations - 1 else cell_value
        row += ";\n"
        A += row
    
    B = ""
    for i in range(nr_equations):
        cell_value = request.form.get(f'matrix_{i}_{nr_variables}')
        B += cell_value + " " if i < nr_equations - 1 else cell_value
    B += ";"
    
    return redirect(url_for('solve_linear_system', A=A, B=B))

if __name__ == '__main__':
   app.run(debug=True, host="0.0.0.0", port=5001)
