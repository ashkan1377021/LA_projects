import numpy as np


def operator(A, k, n, base_vari):
    flag = 0
    r, c = A.shape
    for j in range(k, c - 1):
        for i in range(k, r):
            if (A[i, j] != 0):
                flag = 1
                break
        if flag == 1:
            base_vari.append(j)
            break
    if flag == 1:
        if i > 0:
            temp = A[i].copy()
            A[i] = A[k]
            A[k] = temp
        A[k] = A[k] / A[k, j]
        if k == 0:
            A[1:] -= A[0] * A[1:, j:j + 1]
        elif k != n - 1:
            A[:k, k:] -= A[k, k:] * A[:k, j:j + 1]
            A[k + 1:, k:] -= A[k, k:] * A[k + 1:, j:j + 1]
        else:
            A[:k, k:] -= A[k, k:] * A[:k, j:j + 1]
    if flag == 0:
        return 1
    else:
        return 0


def solve_system_equations(A, base_vari, n, m):
    for i in range(n):
        for j in range(m - 1):
            flag = 0
            if (A[i, j] != 0):
                flag = 1
                break
        if (flag == 0):
            if (A[i][m - 1] != 0):
                # print("doesn't have answer.")
                return
    if (len(base_vari) == m - 1):
        for i in range(m - 1):
            if float(A[i, m - 1]) - int(float(A[i, m - 1])) == 0:
                print("X" + str(i + 1) + " =" + " " + str(int(float(A[i, m - 1]))))
            else:
                print("X" + str(i + 1) + " =" + " " + str(A[i, m - 1]))
    else:
        for k in range(m - 1):
            if not (k in base_vari):
                print("X" + str(k + 1) + " =" + " " + "10")
            else:
                for i in range(n):
                    if (A[i, k] != 0):
                        sum = 0
                        for j in range(m - 1):
                            if j == k:
                                sum -= 0
                            else:
                                sum = sum - A[i, j] * 10
                        sum += A[i, m - 1]
                        if float(sum) - int(float(sum)) == 0:
                            print("X" + str(k + 1) + " =" + " " + str(int(float(sum))))
                        else:
                            print("X" + str(k + 1) + " =" + " " + str(sum))
                        break


n, m = input().split()
n = int(n)
m = int(m)
A = np.zeros((n, m))
for i in range(n):
    A[i, :m] = input().split()
base_vari = list()
flg = 0
for k in range(min(n, m)):
    flg += operator(A, k, n, base_vari)
flg2 = 0
for i in range(n):
    if A[i, m - 1] != 0:
        flg2 = 1
        break
if flg != min(n, m) or (flg == min(n, m) and flg2 == 0):
    for i in range(n):
        for j in range(m):
            if float(A[i, j]) - int(float(A[i, j])) == 0:
                print(int(float(A[i, j])), end=" ")
            else:
                print(A[i, j], end=" ")
        print("")

solve_system_equations(A, base_vari, n, m)
