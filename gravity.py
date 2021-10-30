import csv
import plotly.express as px
import pandas as pd

rows = []

with open("bright_stars.csv", "r") as f:
  csvreader = csv.reader(f)
  for row in csvreader: 
    rows.append(row)

headers = rows[0]
headers2 = 'gravity'
star_data_rows = rows[1:]
# print(headers)
# print(star_data_rows[0])

headers[0] = "row_num"

temp_star_data_rows = list(star_data_rows)

for star_data in temp_star_data_rows:
  star_mass = star_data[3]
  if star_mass.lower() == "unknown":
    star_data_rows.remove(star_data)
    continue
  else:
    star_mass_value = star_mass.split(" ")[0]
    # star_mass_ref = star_mass.split(" ")[2]
    star_data[3] = star_mass_value

  star_radius = star_data[4]
  if star_radius.lower() == "unknown":
    star_data_rows.remove(star_data)
    continue
  else:
    star_radius_value = star_radius.split(" ")[0]
    # star_radius_ref = star_radius.split(" ")[2]
    star_data[4] = star_radius_value

star_masses = []
star_radiuses = []
star_names = []

for star_data in star_data_rows:
  star_masses.append(star_data[3])
  star_radiuses.append(star_data[4])
  star_names.append(star_data[1])

star_masses.sort()
star_radiuses.sort()
star_names.sort()

star_gravity = []

for index, name in enumerate(star_names):
  star_radiuses[index] = star_radiuses[index].replace(",", "")
  star_radiuses[index] = star_radiuses[index].replace("?", "")
  star_radiuses[index] = star_radiuses[index].replace("-", "")
  star_radiuses[index] = star_radiuses[index].replace('"', "")
  gravity = (float(star_masses[index])*1.989e+30) / (float(star_radiuses[index])*float(star_radiuses[index])*6.957e+8*6.957e+8) * 6.674e-11
  star_gravity.append(gravity)

# star_data_rows.append(star_gravity)

print(star_gravity)

star_data = []

# for index, data_row in enumerate(star_data_rows):
#     star_data.append(star_data_rows[index] + star_gravity[index])


# fig = px.scatter(x=star_radiuses, y=star_masses)
# fig.show()

headers3 = headers + list(headers2)

# with open("bright_stars2.csv", "a+") as f:
#     csvwriter = csv.writer(f)
#     csvwriter.writerow(headers)
#     csvwriter.writerows(star_data)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

xwcss = []

for index, star_mass in enumerate(star_masses):
    temp_list = [star_radiuses[index], star_mass]
    xwcss.append(temp_list)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init ='k-means++', random_state = 42)
    kmeans.fit(xwcss)
    wcss.append(kmeans.inertia_)

import numpy as np

x = np.array(star_radiuses).astype(float)
y = np.array(star_masses).astype(float)

plt.figure(figsize=(20,10))
sns.scatterplot(x=x,y=y)
# sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('Chart')
plt.xlabel('Radius')
plt.ylabel('Mass')

# plt.show()

goldilock_range_stars = list(star_data_rows)

# for star_data in star_data_rows:
#   if star_data[2] < 100:
#     goldilock_range_stars.remove(star_data)

gravity_range_stars = list(star_data_rows)
  
for index, star_data in enumerate(star_data_rows):
  if star_gravity[index] < 150 or star_gravity[index] > 350:
    gravity_range_stars.remove(star_data)
  
with open('finalstars.csv', 'a+') as f:
  writer = csv.writer(f)
  writer.writerow(headers)
  writer.writerows(gravity_range_stars)
