import laspy
import open3d as o3d
import numpy as np
import os

#Wczytanie chmury punktów w formacie las
def wczytanie_chmury_punktow_las_konwwersj_do_o3d(plik):
    las_pcd = laspy.read(plik)
    x = las_pcd.x
    y = las_pcd.y
    z = las_pcd.z
#Normalizacja koloru
    r = las_pcd.red/max(las_pcd.red)
    g = las_pcd.green/max(las_pcd.green)
    b = las_pcd.blue/max(las_pcd.blue)
#Konwersja do format NumPy do o3d
    las_points = np.vstack((x,y,z)).transpose()
    las_colors = np.vstack((r,g,b)).transpose()
    chmura_punktow = o3d.geometry.PointCloud()
    chmura_punktow.points = o3d.utility.Vector3dVector(las_points)
    chmura_punktow.colors = o3d.utility.Vector3dVector(las_colors)

    return chmura_punktow


def wyznaczanie_obserwacji_odstających(chmura_punktow, liczba_sasiadow=30, std_ratio=2.0):
    # Usuwanie punktów odstających
    chmura_punktow_odfiltrowana, ind = chmura_punktow.remove_statistical_outlier(liczba_sasiadow, std_ratio)
    # Pobranie punktów odstających (odrzutów) w nowy sposób
    punkty_odstające = chmura_punktow.select_by_index(ind, invert=True)
    return chmura_punktow_odfiltrowana, punkty_odstające

#Downsampling chmur punktów
def regularyzacja_chmur_punktow(chmura_punktów, odleglosc_pomiedzy_wokselami = 0.1):
    chmura_punktów_woksele = chmura_punktów.voxel_down_sample(voxel_size=odleglosc_pomiedzy_wokselami)
    print(f"Wyświetlanie chmury punktów w regularnej siatce wokseli - odległość pomiedzy wokselami: {odleglosc_pomiedzy_wokselami}")
    return chmura_punktów_woksele


files_path = './DaneProjekt/skany_las'
chmura_suma = o3d.geometry.PointCloud()
oberwacje_odstające = o3d.geometry.PointCloud()
for file in os.listdir(files_path):
    if file.endswith('.las'):
        chmura = wczytanie_chmury_punktow_las_konwwersj_do_o3d(os.path.join(files_path, file))
        chmura_punktow_odfiltrowana, punkty_odstające = wyznaczanie_obserwacji_odstających(chmura)
        chmura_suma += chmura_punktow_odfiltrowana
        oberwacje_odstające += punkty_odstające

f = './DaneProjekt/dim_dense_cloud.las'
chmura = wczytanie_chmury_punktow_las_konwwersj_do_o3d(f)
chmura_punktow_odfiltrowana, punkty_odstające = wyznaczanie_obserwacji_odstających(chmura)
chmura_suma += chmura_punktow_odfiltrowana
oberwacje_odstające += punkty_odstające

# save to odstajace.laz and odfiltrowane.laz
o3d.io.write_point_cloud('odstajace.pcd', oberwacje_odstające)
o3d.io.write_point_cloud('odfiltrowane.pcd', chmura_suma)
woksele = regularyzacja_chmur_punktow(chmura_suma)
o3d.io.write_point_cloud('woksele.pcd', woksele)


