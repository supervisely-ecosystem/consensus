from supervisely import Application
from src.ui import layout
import src.globals as g


app = Application(layout=layout, session_info_solid=True)
