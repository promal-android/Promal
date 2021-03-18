import socket, os, time, requests, json

HOOK_APP_REMOTE_ADDR = 9998
outputdir = 'paladin/'
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)
if not os.path.isdir(os.path.join(outputdir,'XML')):
    os.mkdir(os.path.join(outputdir,'XML'))
if not os.path.isdir(os.path.join(outputdir,'Screenshot')):
    os.mkdir(os.path.join(outputdir,'Screenshot'))
if not os.path.isdir(os.path.join(outputdir,'Log')):
    os.mkdir(os.path.join(outputdir,'Log'))

def initserver():
    os.popen("adb shell am instrument -w com.github.uiautomator.test/android.support.test.runner.AndroidJUnitRunner")
    os.popen("adb forward tcp:7654 tcp:9008")

def CheckDirs(pkg):
    if not os.path.isdir(os.path.join(outputdir,'XML',pkg)):
        os.mkdir(os.path.join(outputdir,'XML',pkg))
    if not os.path.isdir(os.path.join(outputdir,'Screenshot',pkg)):
        os.mkdir(os.path.join(outputdir,'Screenshot',pkg))

def postjson(pkg, timetag):
    try:
        data = {"jsonrpc": "2.0", "method": "dumpWindowHierarchy", "id": 1, "params": [False, "view.xml"]}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url='http://127.0.0.1:7654/jsonrpc/0', headers=headers, data=json.dumps(data))
        a = json.loads(response.text)['result'].replace('\r\n', '\n')
        with open(os.path.join(outputdir,'XML',pkg,timetag+'.xml'), "w+", encoding='utf-8') as f:
            f.write(a)
    except:
        print("check uiautomator server")

def SaveCurrentState(pkg, timetag):
    time.sleep(1)#change accordingly
    CheckDirs(pkg)
    os.popen('adb shell screencap -p > '+os.path.join(outputdir,'Screenshot',pkg,timetag+'.png'))
    postjson(pkg, timetag)
    print("OK")
    # os.system('adb shell uiautomator dump /sdcard/'+timetag+'.xml')
    # os.system('adb pull /sdcard/'+timetag+'.xml '+ os.path.join(outputdir,'XML',pkg,timetag+'.xml'))

# initserver()
host = ''#void means starting server on localhost
port = HOOK_APP_REMOTE_ADDR
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1000)
while True:
    conn, addr = s.accept()
    data = conn.recv(10240)
    if not data: continue
    message = repr(data)[2:-1]
    # print(message)
    if 'SaveCurrentState' in message:
        print("Waiting:")
        splits = message.split(',')
        pkg = splits[1]
        timetag = splits[2]
        import threading
        listen_thread = threading.Thread(target=SaveCurrentState, args =(pkg, timetag))
        # listen_thread.setDaemon(True)
        listen_thread.start()
    else:
        print(message)
        pkg = message.split('\\n')[0]
        with open(os.path.join(outputdir,'Log',pkg+'.log'),"a+" , encoding='utf-8') as f:
            f.write(message.replace((pkg+'\\n'),"",1)+'\n')