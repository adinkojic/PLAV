"""Pilot Joystick for Real-Time manaul flying"""

from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

class PilotJoystick(QtWidgets.QMainWindow):
    """Pilot joystick and pause button class for simulation
    pausing_function: func that when called pauses/unpause sim"""

    def __init__(self, pausing_function, aircraft_event):
        super().__init__()
        self.setWindowTitle('Real Time Flying')
        # Create a central widget and layout
        central_widget = QtWidgets.QWidget()
        instrument_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        controls_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)
        instrument_widget.setLayout(controls_layout)
        self.setCentralWidget(instrument_widget)

        # Create plot area using pyqtgraph GraphicsLayoutWidget
        plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(plot_widget)

        # Create control panel with Pause/Unpause buttons
        button_layout = QtWidgets.QHBoxLayout()
        pause_button = QtWidgets.QPushButton("Pause/Play")
        event_button = QtWidgets.QPushButton("Send Event")
        self.joystick = pg.JoystickButton()
        self.joystick.setFixedWidth(30)
        self.joystick.setFixedHeight(30)
        button_layout.addWidget(pause_button)
        button_layout.addWidget(event_button)
        button_layout.addWidget(self.joystick)
        controls_layout.addLayout(button_layout)

        # Connect buttons to simulation control
        pause_button.clicked.connect(pausing_function)
        event_button.clicked.connect(aircraft_event)
        self.show()

    def get_joystick_pos(self):
        """Gets the joystick state"""
        return self.joystick.getState()
