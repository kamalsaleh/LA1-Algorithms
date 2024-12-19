import numpy
import sympy
from flask import (Flask, redirect, render_template, request, url_for)
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, StringField, IntegerField, SelectField
from wtforms.validators import DataRequired, ValidationError
from algorithms import *

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
        
        output_as_text = convert_matrix_to_string(output[0], ring)
        
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
    elif ring_data == "Rationals":
      ring = QQ
    elif ring_data == "FiniteField":
      ring = GF(char)
    elif ring_data == "PolynomialRing":
      domain_data = form.DOMAIN.data
      if domain_data == "Rationals":
        ring, _ = sympy.ring("x", domain=QQ)
      elif domain_data == "FiniteField":
        ring, _ = sympy.ring("x", domain=GF(char))
    else:
      raise ValueError("The ring must be either 'Integers', 'Rationals', 'FiniteField' or 'PolynomialRing'")
    
    A = parse_matrix_from_string(form.A.data, ring)
    B = parse_matrix_from_string(form.B.data, ring)
    
    if A.shape[1] != B.shape[1]:
      return render_template('solve-linear-system.html',
              form=form,
              error="The number of columns in \(A\) and \(B\) must be the same."
            )
    else:
      
      solution_info = solve_left_linear_system(A, B, ring)
      
      return render_template('solve-linear-system.html',
              output=True,
              form=form,
              **solution_info[2])
  
  return render_template('solve-linear-system.html',
              form=form,
            )

@app.route('/compute-inverses', methods=['GET', 'POST'])
def compute_inverses():
    form = MatrixForm()
    if form.validate_on_submit():
        
        ring_data = form.RING.data
        char = form.CHAR.data
        
        if ring_data == "Integers":
          ring = ZZ
        elif ring_data == "Rationals":
          ring = QQ
        elif ring_data == "FiniteField":
          ring = GF(char)
        elif ring_data == "PolynomialRing":
          domain_data = form.DOMAIN.data
          if domain_data == "Rationals":
            ring, _ = sympy.ring("x", domain=QQ)
          elif domain_data == "FiniteField":
            ring, _ = sympy.ring("x", domain=GF(char))
        else:
          raise ValueError("The ring must be either 'Integers', 'Rationals', 'FiniteField' or 'PolynomialRing'")
        
        M = parse_matrix_from_string(form.M.data, ring)
        
        l_invs = left_inverses(M, ring)
        
        if l_invs[0] is not None:
          l_invs_text = convert_matrix_to_string(numpy.vstack([l_invs[0], l_invs[1]]), ring)
        else:
          l_invs_text = "No left inverses found."
        
        r_invs = right_inverses(M, ring)
        
        if r_invs[0] is not None:
          r_invs_text = convert_matrix_to_string(numpy.hstack([r_invs[0], r_invs[1]]), ring)
        else:
          r_invs_text = "No right inverses found."
        
        return render_template('compute-inverses.html',
                    output=True,
                    form=form,
                    l_invs=l_invs[2],
                    l_invs_text=l_invs_text,
                    r_invs=r_invs[2],
                    r_invs_text=r_invs_text
                  )
    else:
        return render_template('compute-inverses.html',
                    output=False,
                    form=form)

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
