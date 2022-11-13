import sim
import sys
import time
import smach
from sympy import sin, cos, pi
import sympy as sym
import numpy as np
from math import *

# ---------------------------------------
# ----------СВЯЗЬ С sim------------------
# ---------------------------------------

sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 2)

if clientID != -1:  # check if client connection successful
    print('Connected to remote API server')
else:
    print('Connection not successful')
    sys.exit('Could not connect')

# ---------------------------------------
# ----------СВЯЗЬ С sim------------------
# ---------------------------------------

Anker_n = [] * 2
Anker_n = ['/Anker_1', '/Anker_2']
er1, descrip_Anker = sim.simxGetObjectHandle(clientID, Anker_n[0], sim.simx_opmode_oneshot_wait)
ant_n = '/Ant'
er2, descrip_ant = sim.simxGetObjectHandle(clientID, ant_n, sim.simx_opmode_oneshot_wait)
print(descrip_Anker, print(descrip_ant), sep='\n')
fi1=1.1
Ro = 200

def Anker_fi_ang():#Получаем угол между передом робота (0 на градусной окружности) и расположением анкера(якоря/маяка), и расстояние между ними
    err1, ank_pos = sim.simxGetObjectPosition(clientID, descrip_Anker, descrip_ant, sim.simx_opmode_oneshot_wait)
    print(err1, ank_pos)
    S = (ank_pos[0]**2+ank_pos[1]**2)**0.5
    f1 = atan2(ank_pos[1], ank_pos[0]) * 180 / pi
    if f1 > 0:
        f1 = 180 - f1
    elif f1 < 0:
        f1 = -180 - f1
    return f1, S
    # print(fi1)


# ant_pos = sim.simxSetObjectPosition(clientID,descrip_ant,descrip_ant,ank_pos,sim.simx_opmode_streaming)
def rote(clientID, rgName): #Отправляем копелии полученный угол и режим ходьбы (стоим/идем)
    res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(clientID, rgName, \
    sim.sim_scripttype_childscript,'AntRotate', [stepMode], [fi1], [], b'', sim.simx_opmode_blocking)

stepMode=1
while True:
    fi1, Ro = Anker_fi_ang()
    print(fi1, Ro)
    if Ro < 0.025:
        stepMode = 0
        print('stepmode = ',stepMode,'; fi1 = ',fi1,'; Ro = ',Ro, sep='')
        fi1 = 0
        rote(clientID, 'Ant')
        break
    if abs(fi1) < 5:
        fi1 = 0
    rote(clientID, 'Ant')
    time.sleep(0.01)





