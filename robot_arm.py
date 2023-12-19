import numpy as np
import re
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import time

fig = plt.figure('main')
fig.subplots_adjust(
    top=1.0,
    left=0.0,
    bottom=0.0,
    right=1.0,
    hspace=0.2,
    wspace=0.2
)

ax = fig.add_subplot(1, 1, 1, projection='3d')  # type:Axes3D.Axes3D
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(-1, 1, 0.2))
ax.set_zticks(np.arange(0, 1, 0.1))
ax.autoscale(False)


class ArmBase():
    def __init__(self, x=0, y=0, z=0) -> None:
        self.pos = np.array([[1, 0, 0, x],
                             [0, 1, 0, y],
                             [0, 0, 1, z],
                             [0, 0, 0, 1]])

    def render(self):
        ax.scatter(self.pos[0][3], self.pos[1][3], self.pos[2][3], s=100)


class ArmNode():
    def __init__(self, theta_z, dis_z, theta_x, dis_x) -> None:
        self.pos = np.eye(4)
        self.DH = np.array([theta_z, dis_z, theta_x, dis_x], dtype=np.float32)
        self.DH_copy = np.copy(self.DH)

    def setRad(self, theta):
        self.DH[0] = theta

    def setAng(self, theta):
        self.DH[0] = theta * np.pi / 180
        # self.DH[0] = self.DH_copy[0] + theta * np.pi / 180

    def forward(self, base=None):
        if base is None:
            base = np.eye(4)
        self.pos = base
        cz = np.cos(self.DH[0])
        sz = np.sin(self.DH[0])
        Tz = np.array([[cz, -sz, 0, 0],
                       [sz, cz, 0, 0],
                       [0, 0, 1, self.DH[1]],
                       [0, 0, 0, 1]])

        cx = np.cos(self.DH[2])
        sx = np.sin(self.DH[2])
        Tx = np.array([[1, 0, 0, self.DH[3]],
                       [0, cx, -sx, 0],
                       [0, sx, cx, 0],
                       [0, 0, 0, 1]])
        T = np.dot(Tz, Tx)
        r = np.dot(base, T)
        r = np.round(r, 3)
        return r

    def render(self):
        # draw the arm
        p = np.dot(self.pos, np.array([0, 0, 0.1, 1]).T)
        s = np.stack([self.pos[0:3, 3], p[0:3]])
        ax.plot(s[:, 0], s[:, 1], s[:, 2])
        # draw the origin
        ax.scatter(self.pos[0][3], self.pos[1][3], self.pos[2][3])


class GCodeBase():
    def __init__(self) -> None:
        self.pendown = False
        # self.rx = 0
        # self.ry = 0
        # self.rz = 0
        # self.x = 0
        # self.y = 0
        # self.z = 0
        self.T = np.zeros(6)
        self.commands = ""
        self.cmd_index = 0
        self._new_t = None
        self._times = None
        self._delay = None

    def load(self, path):
        with open(path, 'r') as f:
            self.commands = f.read().splitlines()

    def GNext(self):
        if self.cmd_index < len(self.commands):
            cmd = self.commands[self.cmd_index]
            self.cmd_index += 1
            re.sub('\s+', ' ', cmd)
            tokens = cmd.split(' ')
            code = tokens[0]
            if code == 'G00':
                self.G00(tokens[1:])
            elif code == 'G01':
                self.G01(tokens[1:])
            elif code == 'G02':
                self.G02(tokens[1:])
            elif code == 'G10':
                self.G10()
            elif code == 'G11':
                self.G11()
        if self.cmd_index < len(self.commands):
            return False
        else:
            return True

    def G00(self, args):
        for i in args:
            if i[0] == 'X':
                x = float(i[1:])
            elif i[0] == 'Y':
                y = float(i[1:])
            elif i[0] == 'Z':
                z = float(i[1:])
            elif i[0] == 'T':
                t = int(i[1:])
            elif i[0] == 'D':
                d = float(i[1:])
            else:
                raise "GCode Wrong"
        _t = self.T.copy()
        _t[3] = x
        _t[4] = y
        _t[5] = z
        self._new_t = _t
        self._times = t
        self._delay = d

    def G01(self, args):
        for i in args:
            if i[0:2] == 'RX':
                rx = float(i[2:])
            elif i[0:2] == 'RY':
                ry = float(i[2:])
            elif i[0:2] == 'RZ':
                rz = float(i[2:])
            elif i[0] == 'T':
                t = int(i[1:])
            elif i[0] == 'D':
                d = float(i[1:])
            else:
                raise "GCode Wrong"
        _t = self.T.copy()
        _t[0] = rx * np.pi / 180
        _t[1] = ry * np.pi / 180
        _t[2] = rz * np.pi / 180
        self._new_t = _t
        self._times = t
        self._delay = d

    def G02(self, args):
        for i in args:
            if i[0:2] == 'RX':
                rx = float(i[2:])
            elif i[0:2] == 'RY':
                ry = float(i[2:])
            elif i[0:2] == 'RZ':
                rz = float(i[2:])
            elif i[0] == 'X':
                x = float(i[1:])
            elif i[0] == 'Y':
                y = float(i[1:])
            elif i[0] == 'Z':
                z = float(i[1:])
            elif i[0] == 'T':
                t = int(i[1:])
            elif i[0] == 'D':
                d = float(i[1:])
            else:
                raise "GCode Wrong"
        _t = self.T.copy()
        _t[0] = rx * np.pi / 180
        _t[1] = ry * np.pi / 180
        _t[2] = rz * np.pi / 180
        _t[3] = x
        _t[4] = y
        _t[5] = z
        self._new_t = _t
        self._times = t
        self._delay = d

    def G10(self):
        self.pendown = True
        self._new_t = None
        self._times = None
        self._delay = None

    def G11(self):
        self.pendown = False
        self._new_t = None
        self._times = None
        self._delay = None


class Arm():
    def __init__(self, x=0, y=0, z=0) -> None:
        self.path = []
        self.g = GCodeBase()
        self.base = ArmBase(x, y, z)
        self.node0 = ArmNode(0, 0.7, -np.pi/2, 0)
        self.node1 = ArmNode(0, 0, 0, 0.5)
        self.node2 = ArmNode(np.pi/2, 0, np.pi/2, 0)
        self.node3 = ArmNode(0, 0.6, -np.pi/2, 0)
        self.node4 = ArmNode(0, 0, np.pi/2, 0)
        self.node5 = ArmNode(0, 0.4, 0, 0)
        self.nodes = [self.node0,
                      self.node1,
                      self.node2,
                      self.node3,
                      self.node4,
                      self.node5]
        self.final_r = np.eye(4)

    def render(self):
        for node_i in self.nodes:
            node_i.render()
        if self.final_r is not None:
            ax.scatter(self.final_r[0][3], self.final_r[1][3], self.final_r[2][3])
            if self.g.pendown:
                self.path.append((self.final_r[0][3], self.final_r[1][3], self.final_r[2][3]))
        for i in range(1, len(self.path)):
            ax.plot([self.path[i-1][0], self.path[i][0]],
                    [self.path[i-1][1], self.path[i][1]],
                    [self.path[i-1][2], self.path[i][2]], c='red')

    def forward(self):
        self.base.render()
        r = self.base.pos
        for node_i in self.nodes:
            r = node_i.forward(r)
        self.final_r = r
        print(r)

    def backward(self, r_x=None, r_y=None, r_z=None, x=None, y=None, z=None):
        t = np.eye(4)
        s = np.sin(r_x)
        c = np.cos(r_x)
        trans_x = np.array([[1, 0, 0, 0],
                            [0, c, -s, 0],
                            [0, s, c, 0],
                            [0, 0, 0, 1]])
        s = np.sin(r_y)
        c = np.cos(r_y)
        trans_y = np.array([[c, 0, -s, 0],
                            [0, 1, 0, 0],
                            [s, 0, c, 0],
                            [0, 0, 0, 1]])
        s = np.sin(r_z)
        c = np.cos(r_z)
        trans_z = np.array([[c, -s, 0, 0],
                            [s, c, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        t = np.dot(t, trans_x)
        t = np.dot(t, trans_y)
        t = np.dot(t, trans_z)
        t[0, 3] = x - self.base.pos[0, 3]
        t[1, 3] = y - self.base.pos[1, 3]
        t[2, 3] = z - self.base.pos[2, 3]
        # t = np.array([[-0.5173, -0.1592, -0.8409, -0.3390],
        #               [0.8335, 0.1290, -0.5372, -0.2153],
        #               [0.1940, -0.9788, 0.0660, 1.5074],
        #               [0, 0, 0, 1]])
        # RX=90 RY=-90 RZ=90
        # t = np.array([[0., 0., 1., 1.0],
        #               [0., 1., 0., -0.5],
        #               [-1., 0., 0., 1.2],
        #               [0., 0., 0., 1.]])
        # position in {4}
        # print(t)
        t4 = t[:, 3:4] - self.node5.DH[1] * t[:, 2:3]
        temp_1 = np.power(t[0, 3] - self.node5.DH[1] * t[0, 2], 2)
        temp_2 = np.power(t[1, 3] - self.node5.DH[1] * t[1, 2], 2)
        temp_3 = np.power(t[2, 3] - self.node5.DH[1] * t[2, 2] - self.node0.DH[1], 2)
        temp_4 = np.power(self.node1.DH[3], 2)
        temp_5 = np.power(self.node3.DH[1], 2)
        temp_6 = temp_1 + temp_2 + temp_3 - temp_4 - temp_5
        temp_7 = 2*self.node1.DH[3]*self.node3.DH[1]
        theta_3 = np.arcsin(temp_6 / temp_7)
        # theta_3 = np.pi - np.arcsin(temp_6 / temp_7)

        temp_A = self.node1.DH[3] + self.node3.DH[1] * np.sin(theta_3)
        temp_B = -self.node3.DH[1] * np.cos(theta_3)
        temp_C = -(t[2, 3] - self.node5.DH[1]*t[2, 2] - self.node0.DH[1])
        temp_fai = np.arctan2(temp_B, temp_A)
        theta_2 = np.arcsin(temp_C / np.sqrt((temp_A ** 2+temp_B ** 2))) - temp_fai
        # theta_2 = np.pi - np.arcsin(temp_C / np.sqrt((temp_A ** 2+temp_B ** 2))) - temp_fai

        temp_c1 = (t[0, 3] - self.node5.DH[1]*t[0, 2]) / (self.node1.DH[3]*np.cos(theta_2) + self.node3.DH[1] * np.sin(theta_2 + theta_3))
        temp_s1 = (t[1, 3] - self.node5.DH[1]*t[1, 2]) / (self.node1.DH[3]*np.cos(theta_2) + self.node3.DH[1] * np.sin(theta_2 + theta_3))
        theta_1 = np.arctan2(temp_s1, temp_c1)

        self.node0.setRad(theta_1)
        self.node1.setRad(theta_2)
        self.node2.setRad(theta_3)
        r = self.node0.forward(self.base.pos)
        r = self.node1.forward(r)
        r = self.node2.forward(r)
        _3R6 = np.dot(np.transpose(r[:3, :3]), t[:3, :3])
        theta_5 = np.arccos(_3R6[2, 2])
        # theta_5 = -np.arccos(_3R6[2, 2])
        theta_4 = np.arctan2(_3R6[1, 2]*np.sin(theta_5), _3R6[0, 2]*np.sin(theta_5))
        theta_6 = np.arctan2(_3R6[2, 1]*np.sin(theta_5), -_3R6[2, 0]*np.sin(theta_5))
        self.node3.setRad(theta_4)
        self.node4.setRad(theta_5)
        self.node5.setRad(theta_6)
        r = self.node3.forward(r)
        r = self.node4.forward(r)
        r = self.node5.forward(r)
        self.final_r = r
        # print(r)
        # print('State: X={:.3f} Y={:.3f} Z={:.3f}'.format(r[0, 3], r[1, 3], r[2, 3]))
        # print(theta_1 * 180 / np.pi)
        # print(theta_2 * 180 / np.pi)
        # print(theta_3 * 180 / np.pi)
        # print(theta_4 * 180 / np.pi)
        # print(theta_5 * 180 / np.pi)
        # print(theta_6 * 180 / np.pi)


class GCodeManager():
    def __init__(self) -> None:
        self.arms = []
        self.gsta = []

    def add(self, arm: Arm):
        self.arms.append(arm)
        self.gsta.append(False)

    def loop(self, arm):
        if arm.g._new_t is not None:
            d = (arm.g._new_t - arm.g.T) / arm.g._times
            for _ in range(arm.g._times):
                ax.clear()
                ax.set_xticks(np.arange(0, 2, 0.2))
                ax.set_yticks(np.arange(-1, 1, 0.2))
                ax.set_zticks(np.arange(0, 2, 0.2))
                ax.autoscale(False)
                arm.g.T += d
                arm.backward(arm.g.T[0], arm.g.T[1], arm.g.T[2], arm.g.T[3], arm.g.T[4], arm.g.T[5])
                print('State:RX={:.3f} RY={:.3f} RZ={:.3f} X={:.3f} Y={:.3f} Z={:.3f}'.format(float(arm.g.T[0]),
                                                                                              float(arm.g.T[1]),
                                                                                              float(arm.g.T[2]),
                                                                                              float(arm.g.T[3]),
                                                                                              float(arm.g.T[4]),
                                                                                              float(arm.g.T[5])))
                for a in self.arms:
                    a.render()
                plt.pause(arm.g._delay)

    def do(self):
        while False in self.gsta:
            for i in range(len(self.gsta)):
                if not self.gsta[i]:
                    self.gsta[i] = self.arms[i].g.GNext()
                    self.loop(self.arms[i])


arm_l = Arm(0, 0.5, 0)
arm_l.g.load('heart_arm_l.g')

arm_r = Arm(0, -0.5, 0)
arm_r.g.load('heart_arm_r.g')

arm_l.render()
gm = GCodeManager()
gm.add(arm_l)
gm.add(arm_r)
gm.do()

plt.show()
