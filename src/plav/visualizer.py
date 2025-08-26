import vtk
from pymavlink import mavutil

# Connect to SITL over UDP (Idk if it works with mission planner running bc of port conflict)
master = mavutil.mavlink_connection('udp:0.0.0.0:14550')
print("Waiting for heartbeat...")
hb = master.wait_heartbeat()
print(f"Heartbeat from system {hb.get_srcSystem()}, component {hb.get_srcComponent()}, type {hb.type}")
if int(hb.type) == 1: 
    print("Connected to fixed wing aircraft")

#Mavlink provides servo outputs in PWM values (1000-2000uS) so this converts to degrees
def pwm_to_angle(pwm, min_pwm=1000, max_pwm=2000, max_angle=90):
    mid = (min_pwm + max_pwm) / 2
    return (pwm - mid) / (max_pwm - mid) * 2 * max_angle

#----------------------------------- VTK Implementation ----------------------------------
# Load fin
def load_part(stl_file, color=(0.8, 0.8, 0.8)):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    return actor

# Load 4 grid fins
RightFin = load_part("GridFin.stl", color=(0.8, 0.8, 0.8))
LeftFin = load_part("GridFin.stl", color=(0.8, 0.8, 0.8))
RudderFin = load_part("GridFin.stl", color=(0.8, 0.8, 0.8))

# Position fins (example placement)
RightFin.SetPosition(20,  0,  50)   # right side
LeftFin.SetPosition(-20, 0,  50)   # left side
RudderFin.SetPosition(0,  20,  50)   # top

# Orient fins so they face outward
#SetOrientation( ,yaw, )
RightFin.SetOrientation(90, 90, -30)
LeftFin.SetOrientation(90, -90, 30)
RudderFin.SetOrientation(90, 0, 180)


# Renderer
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
style = vtk.vtkInteractorStyleUser()  # minimal interactivity, no panning/zooming
interactor.SetInteractorStyle(style)

renderer.AddActor(RightFin)
renderer.AddActor(LeftFin)
renderer.AddActor(RudderFin)
renderer.SetBackground(0.1, 0.2, 0.4)

# Initialize
interactor.Initialize()
render_window.Render()

#----------------------------------- SITL Implementation ----------------------------------
current_aileron_angle = 0
current_elevator_angle = 0
current_rudder_angle = 0

def timer_callback(obj, event):
    msg = master.recv_match(type='SERVO_OUTPUT_RAW', blocking=False)
    if not msg:
        return
    
    aileron_angle = pwm_to_angle(msg.servo1_raw)
    elevator_angle = pwm_to_angle(msg.servo2_raw)  
    rudder_angle = pwm_to_angle(msg.servo4_raw)

    global current_aileron_angle, current_elevator_angle, current_rudder_angle

    delta_A = aileron_angle - current_aileron_angle
    delta_E = elevator_angle - current_elevator_angle
    delta_R = rudder_angle - current_rudder_angle

    RightFin.RotateZ(delta_A)
    LeftFin.RotateZ(delta_E)
    RudderFin.RotateZ(delta_R)
    current_aileron_angle = aileron_angle
    current_elevator_angle = elevator_angle
    current_rudder_angle = rudder_angle

    render_window.Render()

# This ensures the callback is processed repeatedly in the interactor
interactor.AddObserver("TimerEvent", timer_callback)
interactor.CreateRepeatingTimer(30)  # ~33 fps

#-----------------------------------------------------------------------------------------

# Start GUI loop
interactor.Start()
