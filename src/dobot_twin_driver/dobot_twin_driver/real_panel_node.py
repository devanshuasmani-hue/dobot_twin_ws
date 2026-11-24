"""Tk panel for the PHYSICAL robot (namespace `real`). Needs python3-tk.

Workflow that keeps simulation and hardware separate:
   1. type a target (mm)  ->  [Preview in SIM]  the blue simulated twin moves there (nothing real happens)
   2. check it in RViz    ->  tick ARM, then [Execute ...] and confirm -> the ORANGE physical twin follows the real robot
Every real command goes through the guarded core (calibration, ARM, limits, floor margin, fresh data).
"""
import math
import tkinter as tk
from tkinter import messagebox, ttk

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64, Float64MultiArray, String


class RealPanelNode(Node):
    def __init__(self):
        super().__init__('twin_real_panel')
        P = self.create_publisher
        self.arm = P(Bool, 'cmd/arm', 10)
        self.speed = P(Float64, 'cmd/speed', 10)
        self.stop = P(Empty, 'cmd/stop', 10)
        self.home = P(Empty, 'cmd/home', 10)
        self.clear = P(Empty, 'cmd/clear_alarms', 10)
        self.pump_off = P(Empty, 'cmd/pump_off', 10)
        self.goal = P(PointStamped, 'cmd/goal_xyz', 10)
        self.line = P(PointStamped, 'cmd/line_xyz', 10)
        self.grip = P(Float64, 'cmd/gripper', 10)
        self.sim_goal = P(PointStamped, '/sim/cmd/goal_xyz', 10)
        self.sim_line = P(PointStamped, '/sim/cmd/line_xyz', 10)
        self.sim_sync = P(JointState, '/sim/cmd/sync_joints', 10)
        self.status, self.result, self.tcp, self.raw, self.js = '...', '', None, None, None
        self.create_subscription(String, 'twin_status', lambda m: setattr(self, 'status', m.data), 10)
        self.create_subscription(String, 'cmd_result', lambda m: setattr(self, 'result', m.data), 10)
        self.create_subscription(PoseStamped, 'tool_pose', lambda m: setattr(self, 'tcp', (m.pose.position.x, m.pose.position.y, m.pose.position.z)), 10)
        self.create_subscription(Float64MultiArray, 'raw_pose', lambda m: setattr(self, 'raw', list(m.data)), 10)
        self.create_subscription(JointState, 'joint_states', lambda m: setattr(self, 'js', m), 10)

    @staticmethod
    def pt(xyz):
        m = PointStamped()
        m.point.x, m.point.y, m.point.z = xyz
        return m


class Panel:
    def __init__(self, n: RealPanelNode):
        self.n = n
        self.root = tk.Tk()
        self.root.title('Dobot Magician Twin - PHYSICAL ROBOT')
        pad = dict(padx=6, pady=3)
        self.banner = tk.Label(self.root, text='PHYSICAL ROBOT', bg='#e07b00', fg='white', font=('TkDefaultFont', 12, 'bold'))
        self.banner.pack(fill='x')
        self.lbl_status = tk.Label(self.root, text='status: ...', anchor='w', font=('TkDefaultFont', 10, 'bold'))
        self.lbl_status.pack(fill='x', padx=6)
        self.lbl_raw = ttk.Label(self.root, text='robot reports: -', font=('TkFixedFont', 9))
        self.lbl_raw.pack(anchor='w', padx=6)
        self.lbl_tcp = ttk.Label(self.root, text='twin TCP: -', font=('TkFixedFont', 9))
        self.lbl_tcp.pack(anchor='w', padx=6)

        a = ttk.LabelFrame(self.root, text='Safety')
        a.pack(fill='x', **pad)
        self.armed = tk.BooleanVar(value=False)
        ttk.Checkbutton(a, text='ARM motion (auto-expires)', variable=self.armed, command=self._arm).pack(side='left', padx=4)
        self.spd = tk.DoubleVar(value=20)
        ttk.Label(a, text='speed %').pack(side='left')
        tk.Scale(a, from_=1, to=50, orient='horizontal', variable=self.spd, length=120,
                 command=lambda v: self.n.speed.publish(Float64(data=float(v)))).pack(side='left')
        tk.Button(a, text='STOP', bg='#c00000', fg='white', font=('TkDefaultFont', 11, 'bold'),
                  command=self._stop).pack(side='right', padx=6)

        t = ttk.LabelFrame(self.root, text='Target (mm, twin frame)')
        t.pack(fill='x', **pad)
        self.e = {}
        for i, ax in enumerate('XYZ'):
            ttk.Label(t, text=ax).grid(row=0, column=2 * i)
            en = ttk.Entry(t, width=8)
            en.insert(0, '0')
            en.grid(row=0, column=2 * i + 1, padx=3)
            self.e[ax] = en
        ttk.Button(t, text='Use current', command=self._use_current).grid(row=0, column=6, padx=6)
        r2 = ttk.Frame(t)
        r2.grid(row=1, column=0, columnspan=7, pady=4)
        ttk.Button(r2, text='1. Preview in SIM (joint)', command=lambda: self._preview(False)).pack(side='left', padx=2)
        ttk.Button(r2, text='1. Preview in SIM (line)', command=lambda: self._preview(True)).pack(side='left', padx=2)
        r3 = ttk.Frame(t)
        r3.grid(row=2, column=0, columnspan=7, pady=2)
        tk.Button(r3, text='2. EXECUTE on REAL (joint)', bg='#e07b00', fg='white', command=lambda: self._exec(False)).pack(side='left', padx=2)
        tk.Button(r3, text='2. EXECUTE on REAL (line)', bg='#e07b00', fg='white', command=lambda: self._exec(True)).pack(side='left', padx=2)

        m = ttk.LabelFrame(self.root, text='Other')
        m.pack(fill='x', **pad)
        ttk.Button(m, text='Gripper open', command=lambda: self._confirm('open the gripper', lambda: self.n.grip.publish(Float64(data=1.0)))).pack(side='left', padx=2)
        ttk.Button(m, text='Gripper close', command=lambda: self._confirm('close the gripper', lambda: self.n.grip.publish(Float64(data=0.0)))).pack(side='left', padx=2)
        ttk.Button(m, text='Pump OFF', command=lambda: self._confirm('switch the gripper PUMP OFF (anything held will be released)', lambda: self.n.pump_off.publish(Empty()))).pack(side='left', padx=2)
        ttk.Button(m, text='HOME', command=lambda: self._confirm('move the REAL robot to the home pose', lambda: self.n.home.publish(Empty()))).pack(side='left', padx=2)
        ttk.Button(m, text='Clear alarms', command=lambda: self.n.clear.publish(Empty())).pack(side='left', padx=2)
        ttk.Button(m, text='Sync SIM <- REAL', command=self._sync).pack(side='left', padx=8)

        self.lbl_res = tk.Label(self.root, text='', anchor='w', wraplength=420, justify='left')
        self.lbl_res.pack(fill='x', padx=6, pady=4)
        self.root.after(50, self._loop)
        self.root.protocol('WM_DELETE_WINDOW', self._close)

    # ------------------------------------------------------------------
    def _target(self):
        try:
            return tuple(float(self.e[a].get()) / 1000.0 for a in 'XYZ')
        except ValueError:
            messagebox.showerror('Target', 'X, Y, Z must be numbers (mm)')
            return None

    def _use_current(self):
        if self.n.tcp:
            for a, v in zip('XYZ', self.n.tcp):
                self.e[a].delete(0, 'end')
                self.e[a].insert(0, f'{v * 1000:.1f}')

    def _preview(self, line):
        t = self._target()
        if t:
            (self.n.sim_line if line else self.n.sim_goal).publish(self.n.pt(t))

    def _confirm(self, what, fn):
        if messagebox.askokcancel('Confirm REAL robot motion', f'The physical robot will {what}.\n\nIs the workspace clear and your hand on the stop button?'):
            fn()

    def _exec(self, line):
        t = self._target()
        if t:
            self._confirm(f'move to X {t[0] * 1000:.0f}, Y {t[1] * 1000:.0f}, Z {t[2] * 1000:.0f} mm ({"straight line" if line else "joint move"})',
                          lambda: (self.n.line if line else self.n.goal).publish(self.n.pt(t)))

    def _arm(self):
        self.n.arm.publish(Bool(data=bool(self.armed.get())))

    def _stop(self):
        self.n.stop.publish(Empty())
        self.armed.set(False)

    def _sync(self):
        if self.n.js:
            self.n.sim_sync.publish(self.n.js)

    def _close(self):
        self.n.stop.publish(Empty())
        self.root.quit()

    def _loop(self):
        rclpy.spin_once(self.n, timeout_sec=0.0)
        st = self.n.status
        self.lbl_status.config(text='status: ' + st, fg='#106010' if st.startswith('CONNECTED') and 'ALARM' not in st else '#c00000')
        if 'ARMED' not in st:
            self.armed.set(False)
        if self.n.raw:
            x, y, z, r, j1, j2, j3, j4 = self.n.raw
            self.lbl_raw.config(text=f'robot reports: x{x:7.1f} y{y:7.1f} z{z:7.1f} r{r:6.1f} | J {j1:6.1f} {j2:6.1f} {j3:6.1f} {j4:6.1f}')
        if self.n.tcp:
            self.lbl_tcp.config(text='twin TCP (mm): x%7.1f y%7.1f z%7.1f' % tuple(v * 1000 for v in self.n.tcp))
        self.lbl_res.config(text=self.n.result, fg='#106010' if self.n.result.startswith('OK') else '#c00000')
        self.root.after(50, self._loop)

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = RealPanelNode()
    try:
        Panel(node).run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
