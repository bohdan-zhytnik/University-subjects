
def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i = i + 1
            (array[i], array[j]) = (array[j], array[i])
    (array[i + 1], array[high]) = (array[high], array[i + 1])
    return i + 1

def quickSort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
        quickSort(array, low, pi - 1)
        quickSort(array, pi + 1, high)


data = [80,11,45,25,4,3,45,73,17,873,903,24,95,44,752,887,862,530,843,923,876,117,285,191,104,347,291,679,826,8,7613,852,6776,709,4251,4,6,4,4,64,6,523,5,27,6,2,1,3,3,360,2,167,7,4633,55,5,3,241,218,6,94,8883,689,835,6451,758,592,22,158,17,16,6,3,74,1]
quickSort(data, 0, len(data)-1)
print(data)