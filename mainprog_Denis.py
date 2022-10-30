import sim
import sys
import time
import smach
from scipy.optimize import minimize, Bounds
from numpy import sin, cos, pi
import numpy as np

# ---------------------------------------
# ----------СВЯЗЬ С sim-----------------
# ---------------------------------------

sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 2)

if clientID != -1:  # check if client connection successful
    print('Connected to remote API server')
else:
    print('Connection not successful')
    sys.exit('Could not connect')

# err3, vision_sensor = sim.simxGetObjectHandle(clientID, 'Vision_sensor', sim.simx_opmode_oneshot_wait)
# err4, resolution, image = sim.simxGetVisionSensorImage(clientID, vision_sensor, 0, sim.simx_opmode_buffer)


ang_grapp = ['/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/connection/NiryoLGripper', '/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/connection']
# обращение к схвату


ang_joint = [[], []]

ang_joint[0] = ['/NiryoOne/Joint',
                '/NiryoOne/Joint/Link/Joint',
                '/NiryoOne/Joint/Link/Joint/Link/Joint',
                '/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint',
                '/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint',
                '/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint']

ang_joint[1] = ['/youBot/Joint/rollingJoint_fl',
                '/youBot/Joint/rollingJoint_fr',
                '/youBot/rollingJoint_rl',
                '/youBot/rollingJoint_rr'] # колёса YouBot 1,2,3,4 соответственно

d = np.array([0.183, 0, 0, 0.2215, 0, 0.134]) # параметры алгоритма Денавита-Хартенберга
a = np.array([0, 0.21, 0.03, 0, 0, 0.012])
alpha = np.array([-pi / 2, 0, -pi / 2, pi / 2, -pi / 2, 0])
th0 = np.array([0, -pi/2, 0, 0, 0, 0]) # начальные условия для минимизации

forward = [1, 0, 0, -1] # команды для ориентации схвата
left = [0, 1, 1, 0]
right = [0, -1, -1, 0]
back = [-1, 0, 0, 1]

err1, proximity_sensor = sim.simxGetObjectHandle(clientID, '/conveyor/_sensor', sim.simx_opmode_oneshot_wait)

# err2, j2 = sim.simxGetObjectHandle(clientID, ang_joint[i][1], sim.simx_opmode_oneshot_wait)
# err3, j3 = sim.simxGetObjectHandle(clientID, ang_joint[i][2], sim.simx_opmode_oneshot_wait)
# err4, j4 = sim.simxGetObjectHandle(clientID, ang_joint[i][3], sim.simx_opmode_oneshot_wait)
# err5, j5 = sim.simxGetObjectHandle(clientID, ang_joint[i][4], sim.simx_opmode_oneshot_wait)
# err6, j6 = sim.simxGetObjectHandle(clientID, ang_joint[i][5], sim.simx_opmode_oneshot_wait)

# sim.simxSetJointTargetPosition(clientID, j2, pose_arr[1], sim.simx_opmode_streaming)
# sim.simxSetJointTargetPosition(clientID, j3, pose_arr[2], sim.simx_opmode_streaming)
# sim.simxSetJointTargetPosition(clientID, j4, pose_arr[3], sim.simx_opmode_streaming)
# sim.simxSetJointTargetPosition(clientID, j5, pose_arr[4], sim.simx_opmode_streaming)
# sim.simxSetJointTargetPosition(clientID, j6, pose_arr[5], sim.simx_opmode_streaming)


def Manip_Moves(pose_arr, num_manip, DoF): # дексрипторы и команды для шарниров манипулятора
    err = []
    j = []
    for u in range(DoF):
        a, b = sim.simxGetObjectHandle(clientID, ang_joint[num_manip][u], sim.simx_opmode_oneshot_wait)
        err.append(a)
        j.append(b)
        sim.simxSetJointTargetPosition(clientID, j[u], pose_arr[u], sim.simx_opmode_streaming)


def grap(): # дексриптор для схвата
    connection = sim.simxGetObjectHandle(clientID, ang_grapp[0], sim.simx_opmode_oneshot_wait)
    gripper = sim.simxGetObjectChild(clientID, connection[1], 0, sim.simx_opmode_oneshot_wait)
    gripperName = 'NiryoNoGriper'
    if gripper != -1:
        gripperName = sim.simxGetObject


def dhNiryo(TH): # матрицы алгоритма ДХ
    T = np.array([[cos(TH[0]), -sin(TH[0]) * cos(alpha[0]), sin(TH[0]) * sin(alpha[0]), a[0] * cos(TH[0])],
                  [sin(TH[0]), cos(TH[0]) * cos(alpha[0]), -cos(TH[0]) * sin(alpha[0]), a[0] * sin(TH[0])],
                  [0, sin(alpha[0]), cos(alpha[0]), d[0]],
                  [0, 0, 0, 1]])
    for i in 1, 2, 3, 4, 5:
        Tki = np.array([[cos(TH[i]), -sin(TH[i]) * cos(alpha[i]), sin(TH[i]) * sin(alpha[i]), a[i] * cos(TH[i])],
                        [sin(TH[i]), cos(TH[i]) * cos(alpha[i]), -cos(TH[i]) * sin(alpha[i]), a[i] * sin(TH[i])],
                        [0, sin(alpha[i]), cos(alpha[i]), d[i]],
                        [0, 0, 0, 1]])
        T = np.dot(T, Tki)
    return T


def errmin(TH, td): # функция ошибки для задачи минимизации, td матрица однородного
    T = dhNiryo(TH) # преобразования для желаемого положения
    err = 0
    for i in range(2):
        for j in range(2):
            err += (T[i][j]-td[i][j])**2
    err += 7*(T[0][3]-td[0][3])**2
    err += 7*(T[1][3]-td[1][3])**2
    err += 15*(T[2][3]-td[2][3])**2
    return err


def invkinNyryo(x, y, z, e): # обратная задача кинематики с помощью минимизации
    exx, exy, eyx, eyy = e
    Td = [[exx, eyx, 0, x], [exy, eyy, 0, y], [0, 0, -1, z]]
    bnds = Bounds([-35*pi/36, -2.21133, -pi/2, -35*pi/36, -5*pi/9, -2.57436], [35*pi/36, 0, 4*pi/9, 35*pi/36, 11*pi/18, 2.57436])
    res = minimize(errmin, th0, args=Td, method='Nelder-Mead', bounds=bnds, options={'maxiter': 2000, 'maxfev':3000, 'fatol': 0.00001, 'disp': False, 'adaptive': True})
    x1, x2, x3, x4, x5, x6 = res.x
    return [x1, -x2-0.5*pi, -x3, x4, -x5, x6]


# def lsmYouBot(x, y, psi):
#     l = 0.15
#     h = 0.471/2
#     R = 0.05
#     H = np.array([[1, -1, -l-h], [1, 1, l+h], [1, 1, -l-h], [1, -1, l+h]])/R
#     invH = np.dot(np.linalg.inv(np.dot(np.transpose(H), H)),np.transpose(H)) # [H^T.H]^-1.H^T метод МНК
#     dfi =
#     return dfi


def invYouBotVel(VL, VT, Om):
    l = 0.15
    h = 0.471/2
    R = 0.05
    H = np.array([[1, -1, -l-h], [1, 1, l+h], [1, 1, -l-h], [1, -1, l+h]])/R
    v = np.array([VL, VT, Om])
    dfi = np.dot(H, v)
    return dfi


def invYouBotPos(x, y, rot, t):
    l = 0.15
    h = 0.471/2
    R = 0.05
    H = np.array([[1, -1, -l-h], [1, 1, l+h], [1, 1, -l-h], [1, -1, l+h]])/R
    v = np.array([x/t, y/t, rot/t])
    dfi = np.dot(H, v)
    return dfi


def YouBot_Moves(fi):
    err = []
    j = []
    for u in range(4):
        a, b = sim.simxGetObjectHandle(clientID, ang_joint[1][u], sim.simx_opmode_oneshot_wait)
        err.append(a)
        j.append(b)
        sim.simxSetJointTargetVelocity(clientID, j[u], fi[u], sim.simx_opmode_streaming)


def conveyor():
    err, temp, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector= sim.simxReadProximitySensor(clientID, proximity_sensor, sim.simx_opmode_blocking)
    return temp


class State1(smach.State): # запустить конвейер
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome1"])

    def execute(self, ud):
        while conveyor() == 0:
            time.sleep(1)
        return 'outcome1'


class State2(smach.State): # подойти к конвейеру
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome2"])

    def execute(self, ud):
        pose_arr = np.array(invkinNyryo(0.35, 0, 0.11, right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        return 'outcome2'


class State3(smach.State): # схватить писюн (кубик)
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome3"])

    def execute(self, ud):
        sim.simxSetInt32Signal(clientID, 'NiryoLGripper__35___close', 1, sim.simx_opmode_oneshot_wait)
        time.sleep(4)
        return 'outcome3'


class State4(smach.State): # подняться в начально положение
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome4"])

    def execute(self, ud):
        pose_arr = np.array([0, 0, 0, 0, 0, 0], float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        return 'outcome4'


class State5(smach.State): # подойти к другой точке
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome5"])

    def execute(self, ud):
        pose_arr = np.array(invkinNyryo(0, -0.3, 0.01, right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        return 'outcome5'


class State6(smach.State): # открыть схват
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome6"])

    def execute(self, ud):
        sim.simxClearInt32Signal(clientID, 'NiryoLGripper__35___close', sim.simx_opmode_oneshot_wait)
        time.sleep(3)
        return 'outcome6'


class State7(smach.State): # вернуться в начальное положение
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome7"])

    def execute(self, ud):
        pose_arr = np.array([0, 0, 0, 0, 0, 0], float)
        Manip_Moves(pose_arr, 0, 5)
        time.sleep(3)
        return 'outcome7'


class State8(smach.State): # YouBot
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome8"])

    def execute(self, ud):
        YouBot_Moves(invYouBotPos(-1, 0, 2, 5))
        time.sleep(5)
        YouBot_Moves(invYouBotPos(1, 0, 0, 5))
        time.sleep(5)
        YouBot_Moves(invYouBotPos(0, 0, 0, 1))
        return 'outcome8'



def main():
    sm = smach.StateMachine(outcomes=['outcome2'])
    with sm:  # добавление состояний и связей
        # smach.StateMachine.add('State1', State1(), transitions={'outcome1': 'State2'})
        # smach.StateMachine.add('State2', State2(), transitions={'outcome2': 'State3'})
        # smach.StateMachine.add('State3', State3(), transitions={'outcome3': 'State4'})
        # smach.StateMachine.add('State4', State4(), transitions={'outcome4': 'State5'})
        # smach.StateMachine.add('State5', State5(), transitions={'outcome5': 'State6'})
        # smach.StateMachine.add('State6', State6(), transitions={'outcome6': 'State7'})
        # smach.StateMachine.add('State7', State7(), transitions={'outcome7': 'State8'})
        smach.StateMachine.add('State8', State8(), transitions={'outcome8': 'State8'})
        sm.execute()  # запуск машины состояний


if __name__ == '__main__':
    main()
