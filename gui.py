from prodock.vis.dock_gui import DockGUI

# export_static_gui.py
from ipywidgets.embed import embed_minimal_html

gui = DockGUI().build()  # build returns the instance and creates `gui._ui`
widget = gui._ui  # the top-level widget

# write a self-contained HTML file (no Python kernel, no callbacks)
embed_minimal_html("dock_gui_preview.html", views=[widget])
print(
    "Wrote dock_gui_preview.html — open in a browser (no backend callbacks will run)."
)
