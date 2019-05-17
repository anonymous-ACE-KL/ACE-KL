import numpy as np

def flip(idxs, val_ls):
    for idx in idxs:
        if val_ls[idx] == 0:
            val_ls[idx] = 1
        elif val_ls[idx] == 1:
            val_ls[idx] = 0


def main():
    ace_kl_idxs0 = [47,  51,  54,  111, 118, 62,  120, 121, 77, 76]
    ace_kl_idxs1 = [83, 100, 43, 81,  111, 120, 99, 76, 82, 87]
    ace_kl_idxs2 = [83, 100, 81,  43,  111, 120, 99,  76,  82,  49]
    ace_kl_idxs3 = [111, 54, 120, 64, 76, 41, 82, 87, 86, 73]

    lime_idxs0 = [17,  90,  15,  42, 9, 66,  55, 12, 40, 50]
    lime_idxs1 = [17, 15, 16, 9,  40, 66, 106, 12, 42, 90]
    lime_idxs2 = [115, 9, 12,  15,  16, 90, 106,  42,  40,  17]
    lime_idxs3 = [12, 15, 50, 55, 40, 115, 16, 90, 9, 66]


    random_gen = np.random.RandomState(12345)

    '''
    0
    '''
    with open('data/bad0_ori.dat', 'r') as f:
        val_ls0  = map(int, f.readlines()[0].strip('\n').split(' '))
    flip(ace_kl_idxs0, val_ls0)
    with open('data/bad0_amended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls0))+'\n')

    with open('data/bad0_ori.dat', 'r') as f:
        val_ls0  = map(int, f.readlines()[0].strip('\n').split(' '))
    flip(lime_idxs0, val_ls0)
    with open('data/bad0_lamended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls0))+'\n')

    with open('data/bad0_ori.dat', 'r') as f:
        val_ls0  = map(int, f.readlines()[0].strip('\n').split(' '))
    rand_idxs0 = random_gen.choice(range(122), 10)
    flip(rand_idxs0, val_ls0)
    with open('data/bad0_random_amended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls0))+'\n')


    '''
    1
    '''
    with open('data/bad1_ori.dat', 'r') as f:
        val_ls1  = map(int, f.readlines()[0].strip('\n').split(' '))
    flip(ace_kl_idxs1, val_ls1)
    with open('data/bad1_amended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls1))+'\n')

    with open('data/bad1_ori.dat', 'r') as f:
        val_ls1  = map(int, f.readlines()[0].strip('\n').split(' '))
    flip(lime_idxs1, val_ls1)
    with open('data/bad1_lamended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls1))+'\n')

    with open('data/bad1_ori.dat', 'r') as f:
        val_ls1  = map(int, f.readlines()[0].strip('\n').split(' '))
    rand_idxs1 = random_gen.choice(range(122), 10)
    flip(rand_idxs1, val_ls1)
    with open('data/bad1_random_amended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls1))+'\n')

    '''
    2
    '''
    with open('data/bad2_ori.dat', 'r') as f:
        val_ls2  = map(int, f.readlines()[0].strip('\n').split(' '))
    flip(ace_kl_idxs2, val_ls2)
    with open('data/bad2_amended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls2))+'\n')

    with open('data/bad2_ori.dat', 'r') as f:
        val_ls2  = map(int, f.readlines()[0].strip('\n').split(' '))
    flip(lime_idxs2, val_ls2)
    with open('data/bad2_lamended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls2))+'\n')

    with open('data/bad2_ori.dat', 'r') as f:
        val_ls2  = map(int, f.readlines()[0].strip('\n').split(' '))
    rand_idxs2 = random_gen.choice(range(122), 10)
    flip(rand_idxs2, val_ls2)
    with open('data/bad2_random_amended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls2))+'\n')


    '''
    3
    '''
    with open('data/bad3_ori.dat', 'r') as f:
        val_ls3  = map(int, f.readlines()[0].strip('\n').split(' '))
    flip(ace_kl_idxs3, val_ls3)
    with open('data/bad3_amended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls3))+'\n')

    with open('data/bad3_ori.dat', 'r') as f:
        val_ls3  = map(int, f.readlines()[0].strip('\n').split(' '))
    flip(lime_idxs3, val_ls3)
    with open('data/bad3_lamended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls3))+'\n')

    with open('data/bad3_ori.dat', 'r') as f:
        val_ls3  = map(int, f.readlines()[0].strip('\n').split(' '))
    rand_idxs3 = random_gen.choice(range(122), 10)
    flip(rand_idxs3, val_ls3)
    with open('data/bad3_random_amended.dat', 'w') as g:
        g.write('1 122\n')
        g.write(' '.join(map(str, val_ls3))+'\n')

if __name__ == '__main__':
    main()
