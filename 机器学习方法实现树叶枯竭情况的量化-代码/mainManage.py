import os
import featureGet
import xlwt as xls

workbook = xls.Workbook()
worksheet = workbook.add_sheet("sheet1")

i = 0

fileList = os.listdir('../dataset')
for file in fileList:
    try:
        b, g, r, density = featureGet.featureGet(file)
    except ZeroDivisionError:
        continue
    fileID = file.split('.')[0]
    worksheet.write(i, 0, str(fileID))
    worksheet.write(i, 1, str(b))
    worksheet.write(i, 2, str(g))
    worksheet.write(i, 3, str(r))
    worksheet.write(i, 4, str(density))
    i += 1
    print(b)
    print(g)
    print(r)
    print(density)
workbook.save("../feature2.xls")
print("task finished!")
