largest_column_count = 0
csv_file = f'output.csv'
with open(csv_file, 'r') as temp_f:
    lines = temp_f.readlines()
    row_num = len(lines)
    for l in lines:
        column_count = len(l.split(','))
        if largest_column_count < column_count:
            largest_column_count = column_count
temp_f.close()

print(largest_column_count)
