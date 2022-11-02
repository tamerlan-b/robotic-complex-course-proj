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

ang_joint[1] = [''] # массив для другого манипулятора

d = np.array([0.183, 0, 0, 0.2215, 0, 0.134]) # параметры алгоритма Денавита-Хартенберга
a = np.array([0, 0.21, 0.03, 0, 0, 0.012])
alpha = np.array([-pi / 2, 0, -pi / 2, pi / 2, -pi / 2, 0])
th0 = np.array([0, -pi/2, 0, 0, 0, 0]) # начальные условия для минимизации

forward = [1, 0, 0, -1] # команды для ориентации схвата
left = [0, 1, 1, 0]
right = [0, -1, -1, 0]
back = [-1, 0, 0, 1]


# Получаем объект камеры из симулятора
_, conv_cam_handle = sim.simxGetObjectHandle(clientID,'conveyor_camera',sim.simx_opmode_oneshot_wait)

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


def Conveyor_Moves(signal): # попытка подвигать конвейером, полная хуйня
    a, b = sim.simxGetObjectHandle(clientID, 'conveyor', sim.simx_opmode_oneshot_wait)
    if signal == 1:
        sim.simxSetInt32Signal(clientID, 'conveyor_customization-2', 1, sim.simx_opmode_oneshot_wait)
    else:
        sim.simxClearInt32Signal(clientID, 'conveyor_customization-2', sim.simx_opmode_oneshot_wait)


class State1(smach.State): # подойти к конвейеру
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome1"])

    def execute(self, ud):
        Conveyor_Moves(1)
        
        # Получаем координаты кубика 
        # cubes = []
        # errorcode,cam_handle = sim.simxGetObjectHandle(clientID,'conveyor_camera',sim.simx_opmode_oneshot_wait)
        # while(len(cubes) == 0):
        #     image = get_image(conv_cam_handle, clientID)
        #     # Получаем центры объектов
        #     centers = ImageProcessor.findObjects(image)
        #     # Переводим центры в мировую СК
        #     w_centers = [pixel_to_world(c) for c in centers]
        #     # Переводим центры в СК манипулятора
        #     cubes = [world_to_manip(c) for c in w_centers]
        
        # # И передаем их мануипулятору
        # cube = cubes[0]
        # print("Cube coords: ", cube)
        # pose_arr = np.array(invkinNyryo(cube[0], cube[1], cube[2], right), float)
        
        pose_arr = np.array(invkinNyryo(0.35, 0, 0.11, right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        return 'outcome1'


class State2(smach.State): # схватить писюн (кубик)
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome2"])

    def execute(self, ud):
        sim.simxSetInt32Signal(clientID, 'NiryoLGripper__35___close', 1, sim.simx_opmode_oneshot_wait)
        time.sleep(4)
        return 'outcome2'


class State3(smach.State): # подняться в начально положение
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome3"])

    def execute(self, ud):
        pose_arr = np.array([0, 0, 0, 0, 0, 0], float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        return 'outcome3'


class State4(smach.State): # подойти к другой точке
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome4"])

    def execute(self, ud):
        pose_arr = np.array(invkinNyryo(0, -0.3, 0.01, right), float)
        Manip_Moves(pose_arr, 0, 6)
        time.sleep(3)
        return 'outcome4'


class State5(smach.State): # открыть схват
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome5"])

    def execute(self, ud):
        sim.simxClearInt32Signal(clientID, 'NiryoLGripper__35___close', sim.simx_opmode_oneshot_wait)
        time.sleep(3)
        return 'outcome5'


class State6(smach.State): # вернуться в начальное положение
    def __init__(self):
        smach.State.__init__(self, outcomes=["outcome6"])

    def execute(self, ud):
        pose_arr = np.array([0, 0, 0, 0, 0, 0], float)
        Manip_Moves(pose_arr, 0, 5)
        time.sleep(3)
        return 'outcome6'

# Получает RGB-изображение с камеры
def get_image(cam_handle, clientID):
    errorCode,resolution,image=sim.simxGetVisionSensorImage(clientID,cam_handle,0,sim.simx_opmode_streaming)
    if len(image) > 0:
        image = np.array(image,dtype=np.dtype('uint8'))
        image = np.reshape(image,(resolution[1],resolution[0],3))
        # Переводим изображение в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(resolution)
        return image
    else:
        return np.zeros((256, 256, 3))

def pixel_to_world(pixel_pt):
    coeff = 0.000621094
    shift_x = -0.578
    shift_y = -0.27

    u,v = pixel_pt

    wx = shift_x + u * coeff
    wy = shift_y - v * coeff
    wz = 0.1

    return [wx, wy, wz]

def world_to_manip(pt):
    shift_x = -0.35
    shift_y = -0.7
    shift_z = 0.05
    wx,wy,wz = pt
    mx = wx - shift_x
    my = wy - shift_y
    mz = wz - shift_z
    return [mx,my,mz]
    

def main():
    sm = smach.StateMachine(outcomes=['outcome2'])
    with sm:  # добавление состояний и связей
        smach.StateMachine.add('State1', State1(), transitions={'outcome1': 'State2'})
        smach.StateMachine.add('State2', State2(), transitions={'outcome2': 'State3'})
        smach.StateMachine.add('State3', State3(), transitions={'outcome3': 'State4'})
        smach.StateMachine.add('State4', State4(), transitions={'outcome4': 'State5'})
        smach.StateMachine.add('State5', State5(), transitions={'outcome5': 'State6'})
        smach.StateMachine.add('State6', State6(), transitions={'outcome6': 'State1'})
        sm.execute()  # запуск машины состояний


if __name__ == '__main__':
    # Подключение СТЗ
    # errorcode,cam_handle = sim.simxGetObjectHandle(clientID,'conveyor_camera',sim.simx_opmode_oneshot_wait)
    # try:
    #     while sim.simxGetConnectionId(clientID) > -1:
    #         image = get_image(conv_cam_handle, clientID)
    #         # Получаем центры объектов
    #         centers = ImageProcessor.findObjects(image)
    #         # Переводим центры в мировую СК
    #         w_centers = [pixel_to_world(c) for c in centers]
    #         # Переводим центры в СК манипулятора
    #         m_centers = [world_to_manip(c) for c in w_centers]
    #         print(w_centers)
    #         print(m_centers)

    #         # for i,c in enumerate(w_centers):
    #         #     cv2.circle(image, (centers[i][0], centers[i][1]), 7, (0, 0, 0), -1)
    #         #     # text = f"(U,V) = ({c[0]}, {c[1]})"
    #         #     text = f"(x,y,z) = ({c[0]}{c[1]}{c[2]})"
    #         #     text_pos = (centers[i][0] - 30, centers[i][1] - 30)
    #         #     cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    #         # image = ImageProcessor.drawCenters(image, centers)           
    #         cv2.imshow("Conveyor Image", image)
    #         cv2.waitKey(10)
    # except KeyboardInterrupt:   #Checks if ctrl+c is pressed 
    #     pass
    # finally:
    #     cv2.destroyAllWindows()

    main()
