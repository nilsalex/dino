import gi

from screenshare import init_screenshare

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import Gst

screenshare_params = init_screenshare()

print(f"screenshare_params: {screenshare_params}")

Gst.init()

# Pipeline setup
pipeline = Gst.parse_launch(
    f"pipewiresrc fd={screenshare_params.fd} path={screenshare_params.node_id}"
    # "videoconvert ! "
    # "video/x-raw,format=RGBx ! "  # Suggested format
    # "appsink name=sink emit-signals=false sync=false drop=true"
)
# appsink = pipeline.get_by_name("sink")

# # ------------------------------------------------------
# # IMPROVED DYNAMIC CAPS DETECTION
# # ------------------------------------------------------
# width, height, fmt = 1920, 1080, "RGBx"  # Defaults
#
# def initialize_dynamic_caps():
#     global width, height, fmt
#
#     caps_received = threading.Event()
#
#     def on_buffer(pad, info):
#         global width, height, fmt
#
#         caps = pad.get_current_caps()
#         if caps and caps.get_size() > 0:
#             structure = caps.get_structure(0)
#
#             # Extract fields with fallbacks
#             width = structure.get_value("width") or width
#             height = structure.get_value("height") or height
#             fmt = structure.get_value("format") or fmt
#
#             print(f"Detected caps: {width}x{height} ({fmt})")
#             caps_received.set()
#
#         return Gst.PadProbeReturn.REMOVE
#
#     # Add probe to appsink's sink pad
#     sink_pad = appsink.get_static_pad("sink")
#     probe_id = sink_pad.add_probe(
#         Gst.PadProbeType.BLOCKING | Gst.PadProbeType.BUFFER,
#         on_buffer
#     )
#
#     # Start pipeline to trigger buffer flow
#     pipeline.set_state(Gst.State.PLAYING)
#     print("Pipeline started, waiting for caps detection...")
#
#     if not caps_received.wait(5):
#         print("Caps detection timeout - using defaults")
#
#     pipeline.set_state(Gst.State.PAUSED)  # Pause temporarily
#
#     # Remove any remaining probe
#     try:
#         sink_pad.remove_probe(probe_id)
#     except:
#         pass
#
# # ------------------------------------------------------
# # FRAME PROCESSING SYSTEM
# # ------------------------------------------------------
# frame_queue = Queue(maxsize=2)
# exit_event = threading.Event()
#
# def frame_puller():
#     print("Puller thread: Starting")
#
#     # Start pipeline after caps detection setup
#     pipeline.set_state(Gst.State.PLAYING)
#
#     while not exit_event.is_set():
#         sample = appsink.emit("pull-sample")  # Blocking pull
#
#         if not sample:
#             time.sleep(0.001)
#             continue
#
#         buffer = sample.get_buffer()
#         success, map_info = buffer.map(Gst.MapFlags.READ)
#         if success:
#             try:
#                 frame = np.ndarray(
#                     shape=(height, width, 4 if fmt.endswith('x') else 3),
#                     dtype=np.uint8,
#                     buffer=map_info.data
#                 ).copy()  # Copy out of GStreamer buffer
#
#                 frame_queue.put(frame)
#             finally:
#                 buffer.unmap(map_info)
#
#     print("Puller thread: Exiting")
#
# # ------------------------------------------------------
# # MAIN EXECUTION FLOW
# # ------------------------------------------------------
# initialize_dynamic_caps()  # Gets actual caps

# pipeline.set_state(Gst.State.PLAYING)

while True:
    pass

# pull_thread = threading.Thread(target=frame_puller)
# pull_thread.start()
#
# try:
#     frame_count = 0
#     while True:
#         frame = frame_queue.get(timeout=3)
#         frame_count += 1
#         print(f"Processing frame {frame_count} ({frame.shape[1]}x{frame.shape[0]})")
#
# except KeyboardInterrupt:
#     pass
# except Exception as e:
#     print(f"Error: {str(e)}")
# finally:
#     exit_event.set()
#     pull_thread.join(timeout=1.0)
#     pipeline.set_state(Gst.State.NULL)
