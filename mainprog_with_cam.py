import sim
import sys
import time
import smach
from scipy.optimize import minimize, Bounds
from numpy import sin, cos, pi
import numpy as np
import cv2

from image_processor import ImageProcessor

# ---------------------------------------
# ----------СВЯЗЬ С sim-----------------
# ---------------------------------------

sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

if clientID != -1:  # check if client connection successful
    print('Connected to remote API server')
else:
    print('Connection not successful')
    sys.exit('Could not connect')

# Получаем объект камеры из симулятора
_, conv_cam_handle = sim.simxGetObjectHandle(clientID, 'conveyor_camera', sim.simx_opmode_oneshot_wait)

# обращение к схвату
ang_grapp = ['/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/connection/NiryoLGripper', '/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/connection']

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
a = np.array([0, 0.21, 0.03, 0, 0, 0.01])
alpha = np.array([-pi / 2, 0, -pi / 2, pi / 2, -pi / 2, 0])
th0 = np.array([0, -pi/2, 0, 0, 0, pi/2]) # начальные условия для минимизации

forward = [1, 0, 0, -1] # команды для ориентации схвата
left = [0, 1, 1, 0]
right = [0, -1, -1, 0]
back = [-1, 0, 0, 1]

err1, proximity_sensor = sim.simxGetObjectHandle(clientID, '/conveyor/_sensor', sim.simx_opmode_oneshot_wait)


# /////////////////////////NIRYOONE/////////////////////


def Manip_Moves(pose_arr, num_manip, DoF): # дексрипторы и команды для шарниров манипулятора
    err = []
    j = []
    for u in range(DoF):
        a, b = sim.simxGetObjectHandle(clientID, ang_joint[num_manip][u], sim.simx_opmode_oneshot_wait)
        err.append(a)
        j.append(b)
        sim.simxSetJointTargetPosition(clientID, j[u], pose_arr[u], sim.simx_opmode_streaming)
    sim.simxSetObjectFloatParameter(clientID, j[0], 2017, 0.15, sim.simx_opmode_oneshot_wait)
    sim.simxSetObjectFloatParameter(clientID, j[1], 2017, 0.13, sim.simx_opmode_oneshot_wait)


def grap(): # дексриптор для схвата
    connection = sim.simxGetObjectHandle(clientID, ang_grapp[0], sim.simx_opmode_oneshot_wait)
    gripper = sim.simxGetObjectChild(clientID, connection[1], 0, sim.simx_opmode_oneshot_wait)
    gripperName = 'NiryoNoGriper'
    if gripper != -1:
        print('error_gripper')


def dhkin(TH): # матрицы алгоритма ДХ
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
    T = dhkin(TH) # преобразования для желаемого положения
    err = 0
    for i in range(2):
        for j in range(2):
            err += (T[i][j]-td[i][j])**2
    err += 2*(T[0][3]-td[0][3])**2
    err += 2*(T[1][3]-td[1][3])**2
    err += 3*(T[2][3]-td[2][3])**2
    return err


def invkinNyryo(x, y, z, e): # обратная задача кинематики с помощью минимизации
    exx, exy, eyx, eyy = e
    Td = [[exx, eyx, 0, x], [exy, eyy, 0, y], [0, 0, -1, z]]
    bnds = Bounds([-35*pi/36, -2.21133, -pi/2, -35*pi/36, -5*pi/9, -2.57436], [35*pi/36, 0, 4*pi/9, 35*pi/36, 11*pi/18, 2.57436])
    res = minimize(errmin, th0, args=Td, method='Nelder-Mead', bounds=bnds, options={'maxiter': 5000, 'maxfev': 8000, 'fatol': 0.00000001, 'disp': False, 'adaptive': True})
    x1, x2, x3, x4, x5, x6 = res.x
    return [x1, -x2-0.5*pi, -x3, x4, -x5, x6]


# /////////////////////////YOUBOT/////////////////////


def YouBot_Moves(x, y, rot, t):
    err, youbot = sim.simxGetObjectHandle(clientID, '/youBot', sim.simx_opmode_oneshot_wait)
    xr = sim.simxGetObjectPosition(clientID, youbot, -1, sim.simx_opmode_oneshot_wait)[1][0]
    yr = sim.simxGetObjectPosition(clientID, youbot, -1, sim.simx_opmode_oneshot_wait)[1][1]
    rotr = sim.simxGetObjectOrientation(clientID, youbot, -1, sim.simx_opmode_oneshot_wait)[1][2]
    while abs(x-xr) > 0.05 or abs(y-yr) > 0.05 or abs(rot - rotr) > 0.1:
        es1, retInts1, retFloats1, retStrings1, retBuffer1 = sim.simxCallScriptFunction(clientID, '/youBot', sim.sim_scripttype_childscript, 'fast', [], [x, y, rot, t], [], bytearray(), sim.simx_opmode_blocking)
        xr, yr, rotr = retFloats1
        t = t - 0.03
    while abs(x-xr) > 0.001 or abs(y-yr) > 0.001 or abs(rot - rotr) > 0.01:
        es2, retInts2, retFloats2, retStrings2, retBuffer2 = sim.simxCallScriptFunction(clientID, '/youBot', sim.sim_scripttype_childscript, 'slow', [], [x, y, rot, t], [], bytearray(), sim.simx_opmode_blocking)
        xr, yr, rotr = retFloats2
    es3, retInts3, retFloats3, retStrings3, retBuffer3 = sim.simxCallScriptFunction(clientID, '/youBot', sim.sim_scripttype_childscript, 'stop', [], [], [], bytearray(), sim.simx_opmode_blocking)
    print(retStrings3[0])


# /////////////////////////КОНВЕЙЕР/////////////////////


def Conveyor_Moves(signal):  # попытка подвигать конвейером, полная хуйня
    a, b = sim.simxGetObjectHandle(clientID, 'conveyor', sim.simx_opmode_oneshot_wait)
    if signal == 1:
        sim.simxSetInt32Signal(clientID, 'conveyor_customization-2', 1, sim.simx_opmode_oneshot_wait)
    else:
        sim.simxClearInt32Signal(clientID, 'conveyor_customization-2', sim.simx_opmode_oneshot_wait)


# /////////////////////////КАМЕРА/////////////////////


# Получает RGB-изображение с камеры
def get_image(cam_handle, clientID):
    errorCode, resolution, image = sim.simxGetVisionSensorImage(clientID, cam_handle, 0, sim.simx_opmode_streaming)
    if len(image) > 0:
        image = np.array(image, dtype=np.dtype('uint8'))
        image = np.reshape(image, (resolution[1], resolution[0], 3))
        # Переводим изображение в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(resolution)
        return image
    else:
        return np.zeros((256, 256, 3))


def pixel_to_world(pixel_pt):
    coeff = 0.00059055
    shift_x = -0.575
    shift_y = -0.425
    u, v = pixel_pt
    print(pixel_pt)
    wx = shift_x + u * coeff
    wy = shift_y + v * coeff
    wz = 0.1
    return [wx, wy, wz]


def world_to_manip(pt):
    shift_x = 0.35
    shift_y = 0.7
    shift_z = 0.01
    wx, wy, wz = pt
    mx = wy + shift_y
    my = -wx - shift_x
    mz = wz + shift_z
    return [mx, my, mz]


def Cum_scan():
    cubes = []
    errorcode, cam_handle = sim.simxGetObjectHandle(clientID, 'conveyor_camera', sim.simx_opmode_oneshot_wait)
    while len(cubes) == 0:
        image = get_image(conv_cam_handle, clientID)
        centers = ImageProcessor.findObjects(image)
        w_centers = [pixel_to_world(c) for c in centers]
        cubes = [world_to_manip(c) for c in w_centers]
    print("centers: ", centers)
    print("w_centers: ", w_centers)
    return cubes[0]


# //////////////////////СОСТОЯНИЯ/////////////////


class State1(smach.State): # запустить конвейер
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome1"])

    def execute(self, ud):
        Conveyor_Moves(1)
        time.sleep(1)
        return 'outcome1'


class State2(smach.State): # подойти к конвейеру
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome2"])

    def execute(self, ud):
        sim.simxClearInt32Signal(clientID, 'NiryoLGripper__101___close', sim.simx_opmode_oneshot_wait)
        cube = Cum_scan()
        pose_arr = np.array(invkinNyryo(cube[0] - 0.005, cube[1], cube[2] + 0.02, right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(6)
        pose_arr = np.array(invkinNyryo(cube[0] - 0.005, cube[1], cube[2], right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(2)
        return 'outcome2'


class State3(smach.State): # схватить писюн (кубик)
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome3"])

    def execute(self, ud):
        sim.simxSetInt32Signal(clientID, 'NiryoLGripper__101___close', 1, sim.simx_opmode_oneshot_wait)
        time.sleep(5)
        return 'outcome3'


class State4(smach.State): # подняться в начально положение
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome4"])

    def execute(self, ud):
        pose_arr = np.array([0, 0, 0, 0, 0, 0], float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        return 'outcome4'

class State5(smach.State): # юбот едет к манипулятору
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome5"])

    def execute(self, ud):
        YouBot_Moves(0, -0.5, 0, 4)
        time.sleep(10)
        return 'outcome5'


class State6(smach.State): # манипуялтор подходит к другой точке
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome6"])

    def execute(self, ud):
        pose_arr = np.array(invkinNyryo(0, -0.2, 0.25, right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(6)
        pose_arr = np.array(invkinNyryo(0, -0.37, 0.25, right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        pose_arr = np.array(invkinNyryo(0, -0.37, 0.2, right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        return 'outcome6'


class State7(smach.State): # манипулятор отпускает кубик и отходит в сторону
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome7"])

    def execute(self, ud):
        sim.simxClearInt32Signal(clientID, 'NiryoLGripper__101___close', sim.simx_opmode_oneshot_wait)
        time.sleep(3)
        pose_arr = np.array(invkinNyryo(0, -0.2, 0.25, right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        return 'outcome7'


class State8(smach.State): # вернуться в начальное положение
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome8"])

    def execute(self, ud):
        pose_arr = np.array([0, 0, 0, 0, 0, 0], float)
        Manip_Moves(pose_arr, 0, 5)
        time.sleep(3)
        return 'outcome8'


class State9(smach.State): # YouBot
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome9"])

    def execute(self, ud):
        YouBot_Moves(1, 1, -pi / 2, 10)
        time.sleep(3)
        YouBot_Moves(0, 0, 0, 10)
        time.sleep(6)
        return 'outcome9'



def main():
    sm = smach.StateMachine(outcomes=['outcome1'])
    with sm:  # добавление состояний и связей
        smach.StateMachine.add('State1', State1(), transitions={'outcome1': 'State2'})
        smach.StateMachine.add('State2', State2(), transitions={'outcome2': 'State3'})
        smach.StateMachine.add('State3', State3(), transitions={'outcome3': 'State4'})
        smach.StateMachine.add('State4', State4(), transitions={'outcome4': 'State5'})
        smach.StateMachine.add('State5', State5(), transitions={'outcome5': 'State6'})
        smach.StateMachine.add('State6', State6(), transitions={'outcome6': 'State7'})
        smach.StateMachine.add('State7', State7(), transitions={'outcome7': 'State8'})
        smach.StateMachine.add('State8', State8(), transitions={'outcome8': 'State9'})
        smach.StateMachine.add('State9', State9(), transitions={'outcome9': 'State1'})
        sm.execute()  # запуск машины состояний


if __name__ == '__main__':
    main()
