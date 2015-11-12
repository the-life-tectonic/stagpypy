import sys
from subprocess import call,check_output

class NullProgressBar(object):
    def __init__(self):
        pass

    def progress(self,amt):
        pass

class ProgressBar(object):
    def __init__(self,width=-1,total=100,out=sys.stderr):
        if width==-1:
            rows,cols=rows_and_cols()
            self.width=cols-2
        else:
            self.width=int(width)-2
        self.total=float(total)
        self.out=out
        self.fill=0
        out.write('[')
        out.write(' '*self.width)
        out.write(']')

    def progress(self,amt):
        fill=int(self.width*amt/self.total)
        if fill!=self.fill:
            self.out.write('\b'*(self.width+1))
            self.out.write('='*fill)
            self.out.write(' '*(self.width-fill))
            self.out.write(']')
            self.fill=fill
        

class Spinner(object):
    def __init__(self,size=1,out=sys.stderr):
        self.spinner='|'*size + '/'*size + '-'*size + '\\'*size + '|'*size + '/'*size + '-'*size + '\\'*size 
        self.n=-1
        self.out=out
        out.write(self.spinner[0])
        self.out.flush()

    def next(self):
        self.out.write('\b')
        self.n=(self.n+1)%len(self.spinner)
        self.out.write(self.spinner[self.n])
        self.out.flush()

_current_cursor=True
def cursor(mode=None):
    global _current_cursor
    _current_cursor=mode if mode != None else not _current_cursor
    cursor='on' if _current_cursor else 'off'
    return call(['setterm','-cursor',cursor])

def rows_and_cols():                                       
    return [int(n) for n in check_output(['stty','size']).split()]

def test():
    print("The display has %d rows and %d cols"%tuple(rows_and_cols()))
    cursor(False)
    import time
    print("A spinner : ")
    s=Spinner()
    i=0
    while i<60:
        s.next()
        time.sleep(.05)
        i=i+1

    print()
    print('A progress bar : ')
    p=ProgressBar()
    for i in range(101):
        p.progress(i)
        time.sleep(.1)

    print()
    print('A progress bar with a different total : ')
    p=ProgressBar(total=10)
    for i in range(11):
        p.progress(i)
        time.sleep(1)

    print()
    print('A set width progress bar: ')
    p=ProgressBar(width=20)
    for i in range(101):
        p.progress(i)
        time.sleep(.1)
    cursor(True)
    

