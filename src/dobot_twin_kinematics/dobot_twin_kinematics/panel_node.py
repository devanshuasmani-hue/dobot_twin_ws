"""Tk control panel for the SIMULATED twin: Cartesian X/Y/Z, jog, joints readout, gripper.
Runs in namespace `sim` (topics relative). Needs `sudo apt install python3-tk`.
Publishes only to /sim/cmd/* - it can never command a physical robot.
"""
import math
import tkinter as tk
from tkinter import ttk

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64, String

RANGES = {'X': (-0.30, 0.30), 'Y': (-0.30, 0.30), 'Z': (0.0, 0.25)}   # panel slider ranges [m]
GRIP_MAX = 0.0135


class PanelNode(Node):
    def __init__(self):
        super().__init__('twin_control_panel')
        self.pub_goal = self.create_publisher(PointStamped, 'cmd/goal_xyz', 10)
        self.pub_line = self.create_publisher(PointStamped, 'cmd/line_xyz', 10)
        self.pub_jog = self.create_publisher(Twist, 'cmd/jog', 10)
        self.pub_grip = self.create_publisher(Float64, 'cmd/gripper', 10)
        self.pub_home = self.create_publisher(Empty, 'cmd/home', 10)
        self.pub_stop = self.create_publisher(Empty, 'cmd/stop', 10)
        self.tcp = None
        self.joints = {}
        self.status = '...'
        self.create_subscription(PoseStamped, 'tool_pose', self._pose, 10)
        self.create_subscription(JointState, 'joint_states', self._js, 10)
        self.create_subscription(String, 'status', self._status, 10)

    def _pose(self, m):
        self.tcp = (m.pose.position.x, m.pose.position.y, m.pose.position.z)

    def _js(self, m):
        self.joints = dict(zip(m.name, m.position))

    def _status(self, m):
        self.status = m.data

    def goal(self, xyz, line=False):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = xyz
        (self.pub_line if line else self.pub_goal).publish(msg)

    def jog(self, v, w=0.0):
        t = Twist()
        t.linear.x, t.linear.y, t.linear.z = v
        t.angular.z = w
        self.pub_jog.publish(t)


class Panel:
    def __init__(self, node: PanelNode):
        self.n = node
        self.root = tk.Tk()
        self.root.title('Dobot Magician Twin - SIMULATION control')
        self.jogv = (0.0, 0.0, 0.0, 0.0)
        self.synced = False
        pad = dict(padx=6, pady=3)

        banner = tk.Label(self.root, text='SIMULATION  (not connected to any robot)', bg='#2f8cff', fg='white',
                          font=('TkDefaultFont', 12, 'bold'))
        banner.pack(fill='x')

        # --- Cartesian target ---------------------------------------------------------------
        f = ttk.LabelFrame(self.root, text='Target position of the gripper tip [mm]')
        f.pack(fill='x', **pad)
        self.vars, self.sliders = {}, {}
        for i, ax in enumerate('XYZ'):
            lo, hi = RANGES[ax]
            var = tk.DoubleVar(value=0.0)
            self.vars[ax] = var
            ttk.Label(f, text=ax, width=2).grid(row=i, column=0)
            s = tk.Scale(f, from_=lo * 1000, to=hi * 1000, orient='horizontal', length=330, resolution=1, variable=var,
                         command=lambda _v: self._live())
            s.grid(row=i, column=1, **pad)
            self.sliders[ax] = s
        self.live = tk.BooleanVar(value=False)
        row = ttk.Frame(f)
        row.grid(row=3, column=0, columnspan=2, **pad)
        ttk.Button(row, text='Go (joint move)', command=lambda: self._go(False)).pack(side='left', padx=3)
        ttk.Button(row, text='Go (straight line)', command=lambda: self._go(True)).pack(side='left', padx=3)
        ttk.Checkbutton(row, text='live (follow sliders)', variable=self.live).pack(side='left', padx=8)

        # --- Jog ------------------------------------------------------------------------------
        j = ttk.LabelFrame(self.root, text='Jog (hold button)')
        j.pack(fill='x', **pad)
        self.speed = tk.DoubleVar(value=40.0)
        spd = ttk.Frame(j)
        spd.pack(fill='x')
        ttk.Label(spd, text='speed mm/s').pack(side='left')
        tk.Scale(spd, from_=5, to=100, orient='horizontal', variable=self.speed, length=200).pack(side='left')
        grid = ttk.Frame(j)
        grid.pack()
        for c, (lab, vec) in enumerate([('X-', (-1, 0, 0, 0)), ('X+', (1, 0, 0, 0)), ('Y-', (0, -1, 0, 0)),
                                        ('Y+', (0, 1, 0, 0)), ('Z-', (0, 0, -1, 0)), ('Z+', (0, 0, 1, 0)),
                                        ('J4-', (0, 0, 0, -1)), ('J4+', (0, 0, 0, 1))]):
            b = ttk.Button(grid, text=lab, width=5)
            b.grid(row=0, column=c, padx=2, pady=2)
            b.bind('<ButtonPress-1>', lambda _e, v=vec: self._set_jog(v))
            b.bind('<ButtonRelease-1>', lambda _e: self._set_jog((0, 0, 0, 0)))

        # --- gripper / misc --------------------------------------------------------------------
        g = ttk.LabelFrame(self.root, text='Gripper & misc')
        g.pack(fill='x', **pad)
        ttk.Button(g, text='Open', command=lambda: self.n.pub_grip.publish(Float64(data=GRIP_MAX))).pack(side='left', padx=3)
        ttk.Button(g, text='Close', command=lambda: self.n.pub_grip.publish(Float64(data=0.0))).pack(side='left', padx=3)
        ttk.Button(g, text='HOME', command=lambda: self.n.pub_home.publish(Empty())).pack(side='left', padx=12)
        ttk.Button(g, text='STOP', command=lambda: self.n.pub_stop.publish(Empty())).pack(side='left', padx=3)

        # --- readouts ---------------------------------------------------------------------------
        r = ttk.LabelFrame(self.root, text='State')
        r.pack(fill='x', **pad)
        self.lbl_tcp = ttk.Label(r, text='TCP: -', font=('TkFixedFont', 10))
        self.lbl_tcp.pack(anchor='w')
        self.lbl_j = ttk.Label(r, text='Joints: -', font=('TkFixedFont', 10))
        self.lbl_j.pack(anchor='w')
        self.lbl_st = tk.Label(r, text='status: -', anchor='w')
        self.lbl_st.pack(fill='x')

        self.root.after(50, self._loop)
        self.root.protocol('WM_DELETE_WINDOW', self.root.quit)

    # ------------------------------------------------------------------ actions
    def _target(self):
        return tuple(self.vars[a].get() / 1000.0 for a in 'XYZ')

    def _go(self, line):
        self.n.goal(self._target(), line)

    def _live(self):
        if self.live.get() and self.synced:
            self.n.goal(self._target(), False)

    def _set_jog(self, vec):
        self.jogv = tuple(vec)

    # --------------------------------------------------------------------- loop
    def _loop(self):
        rclpy.spin_once(self.n, timeout_sec=0.0)
        if any(self.jogv):
            s = self.speed.get() / 1000.0
            self.n.jog((self.jogv[0] * s, self.jogv[1] * s, self.jogv[2] * s), self.jogv[3] * math.radians(40))
        if self.n.tcp:
            x, y, z = self.n.tcp
            if not self.synced:                       # first pose: initialise sliders without moving the robot
                self.synced = True
                for a, v in zip('XYZ', (x, y, z)):
                    self.vars[a].set(round(v * 1000))
            self.lbl_tcp.config(text=f'TCP  X {x * 1000:7.1f}  Y {y * 1000:7.1f}  Z {z * 1000:7.1f} mm')
        if self.n.joints:
            d = lambda k: math.degrees(self.n.joints.get(k, 0.0))      # noqa: E731
            self.lbl_j.config(text='J1 %6.1f  J2 %6.1f  J3 %6.1f  J4 %6.1f deg   grip %4.1f mm' % (
                d('magician_joint_1'), d('magician_joint_2'), d('magician_joint_3'), d('magician_joint_4'),
                1000 * self.n.joints.get('magician_joint_prismatic_l', 0.0)))
        st = self.n.status
        bad = st.startswith('ERROR') or st.startswith('LIMIT')
        self.lbl_st.config(text='status: ' + st, fg='#c00000' if bad else '#106010')
        self.root.after(50, self._loop)

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = PanelNode()
    try:
        Panel(node).run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
