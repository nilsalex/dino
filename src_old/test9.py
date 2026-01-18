import gi

from screenshare import init_screenshare

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst

screenshare_params = init_screenshare()

print(f"screenshare_params: {screenshare_params}")

Gst.init()

pipeline = Gst.parse_launch(
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id} "
    # "! videoconvert "
    # "! video/x-raw,format=RGBA "
    "! pipewiresink name=ps_out stream-properties=props,media.name=python-screenshare"
)

pipeline.set_state(Gst.State.PLAYING)

loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pipeline.set_state(Gst.State.NULL)
