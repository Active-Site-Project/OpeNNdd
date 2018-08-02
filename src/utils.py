def binSearch(arr, target):
    low = 0
    high = len(arr)
    mid = (low + high)//2
    found = False

    if target < arr[0]:
        return 0

    while (not found):
        if (target < arr[mid] and target >=arr[mid-1]):
            found = True
        elif (target >= arr[mid]):
            low = mid+1
            mid = (low+high)//2
        else:
            high = mid-1
            mid = (low+high)//2
    return mid
