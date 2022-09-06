##functions for the frontend aspect of the website
##store standard roots for the website, namely where users can actually go to (home page)
from flask import Blueprint, render_template
# from flask_login import login_required, current_user

views = Blueprint('views', __name__)

@views.route('/')
# @login_required
##this function will run whenever we goo to this routh or path
def home():
    return render_template('home.html', )


