from flask import Blueprint

portal_bp = Blueprint("portal", __name__)

from . import routes
