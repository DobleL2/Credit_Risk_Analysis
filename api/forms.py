from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Email, Length, NumberRange

class MyForm(FlaskForm):
    id_client = StringField('Client ID', validators=[DataRequired()])
    payment_day = IntegerField('Payment Day', validators=[DataRequired(), NumberRange(min=1, max=31)])
    sex = SelectField('Sex', choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    marital_status = SelectField('Marital Status', choices=[('single', 'Single'), ('married', 'Married')], validators=[DataRequired()])
    quant_dependants = IntegerField('Number of Dependents', validators=[DataRequired(), NumberRange(min=0)])
    nacionality = StringField('Nationality', validators=[DataRequired()])
    flag_residencial_phone = BooleanField('Residential Phone Flag')
    residence_type = StringField('Residence Type', validators=[DataRequired()])
    months_in_residence = IntegerField('Months in Residence', validators=[DataRequired(), NumberRange(min=0)])
    personal_monthly_income = IntegerField('Personal Monthly Income', validators=[DataRequired(), NumberRange(min=0)])
    other_incomes = IntegerField('Other Incomes', validators=[DataRequired(), NumberRange(min=0)])
    quant_banking_accounts = IntegerField('Number of Banking Accounts', validators=[DataRequired(), NumberRange(min=0)])
    personal_assets_value = IntegerField('Personal Assets Value', validators=[DataRequired(), NumberRange(min=0)])
    quant_cars = IntegerField('Number of Cars', validators=[DataRequired(), NumberRange(min=0)])
    falg_professional_phone = BooleanField('Professional Phone Flag')
    profession_code = StringField('Profession Code', validators=[DataRequired()])
    occupation_type = StringField('Occupation Type', validators=[DataRequired()])
    product = StringField('Product', validators=[DataRequired()])
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=0)])
    residencial_zip_3 = StringField('Residential ZIP', validators=[DataRequired(), Length(min=3, max=3)])
    has_any_card = BooleanField('Has Any Card')
    submit = SubmitField('Submit')
