from flask import Flask, render_template
from model.model import *
from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, StringField, validators
from flask import render_template, flash, redirect
import numpy as np
import os


app = Flask("prediksi delay")
app.config['SECRET_KEY'] = "12345678"
model = model()

class config(object):
    KEY = os.environ.get("SECRET_KEY") or "12345678"

app.config.from_object(config)

class value_input(FlaskForm):
    Airline = StringField("Airline", [validators.input_required(), validators.length(max=2)])
    Flight = IntegerField("Flight")
    AirportTo = StringField("AirportTo", [validators.input_required(), validators.length(max=3)])
    Time = IntegerField("Time")
    Length = IntegerField("Length")
    submit = SubmitField("prediksi")

def delay_status(x):
    if(x == 1):
        return "Delay"
    return "Tidak Delay"

@app.route("/")
def form_input():
    form = value_input()
    performance = model.get_model_performance()
    size = model.get_data_size()
    return render_template('main.html', title='prediksi delay', form=form, performance=performance, size=size)

@app.route('/', methods=['GET', "POST"])
def predict():
    form = value_input()
    if(form.validate_on_submit()):
        flash("kategori keadaannya {}".format(delay_status( model.predict( np.array([[
            form.Airline.data, form.Flight.data, form.AirportTo.data, form.Time.data, form.Length.data
        ]], dtype=object) ))) )
        return redirect('/')
    return render_template('main.html', title="prediksi delay", form=form)

if(__name__ == "__main__"):
    app.run(debug=True)
