# Робототехнический комплекс в рамках проекта по "Управлению РТК"  

Робототехнический комплекс, выполненный в рамках проекта по "Управлению РТК" в МГТУ им. Н.Э. Баумана в симуляторе CoppeliaSim.  

### Установка

Для работы симулятора потребуются библиотеки:  
- smach
- scipy
- opencv
- подходящая библиотека с remote API: "remoteApi.dll" (Windows), "remoteApi.dylib" (Mac) или "remoteApi.so" (Linux)

Чтобы установить smach, заходим в каталог smach из командной строки и выполняем инструкцию:  
```bash  
python setup.py install
```
На Linux может потребоваться `sudo`.  

Библиотеки opencv и scipy устанавливаются через pip:  
```bash
pip install opencv-python
pip install scipy
```

Подходящую библиотеку с remote API можно найти в папке с установленной программой CoppeliSim по пути: **programming/legacyRemoteApi/remoteApiBindings/lib/lib/**. Далее нужно перейти в каталог со своей ОС и скопировать файл в папку с проектом.  
Данный проект тестировался на Ubuntu 20.04 и соответствующая бибилотека добавлена
### Структура проекта

Ключевые файлы проекта:  
- mainprog.py - код работы РТК
- image_processor.py - код обработки изображения
- MyProject.ttt - сцена с РТК для симулятора
- smach - каталог с библиотекой
- remoteApi.so - библиотека remoteApi для Ubuntu 20.04  

### Запуск

1. Запустить симулятор CoppeliaSim и открыть в нем файл MyProject.ttt
2. Запустить симуляцию
3. Открыть терминал в папке с проектом и запустить скрипт:  
```bash  
python mainprog.py
```