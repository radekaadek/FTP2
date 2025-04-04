import open3d as o3d
import numpy as np

def ICP_registration(source, target, threshold=1.0, trans_init=np.identity(4), metoda='p2p', max_iteration=100):
    # calculate the centroid of both clouds
    print('Analiza dokładności wstępnej orientacji')
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    conv_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    
    if metoda == 'p2p':
        print("Orientacja ICP <Punkt do punktu>")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=conv_criteria
        )
        print(reg_p2p)
        print("Macierz transformacji:")
        print(reg_p2p.transformation)
        
        information_reg_p2p = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, threshold, reg_p2p.transformation
        )
        return reg_p2p.transformation, information_reg_p2p
    
    elif metoda == 'p2pl':
        print('Wyznaczanie normalnych')
        source.estimate_normals()
        target.estimate_normals()
        
        print("Orientacja ICP <Punkt do płaszczyzny>")
        reg_p2pl = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=conv_criteria
        )
        print(reg_p2pl)
        print("Macierz transformacji:")
        print(reg_p2pl.transformation)
        
        information_reg_p2pl = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, threshold, reg_p2pl.transformation
        )
        return reg_p2pl.transformation, information_reg_p2pl
    
    elif metoda == 'cicp':
        print("Orientacja ICP <Kolorowy ICP>")
        reg_cicp = o3d.pipelines.registration.registration_colored_icp(
            source, target, threshold, trans_init,
            criteria=conv_criteria
        )
        print(reg_cicp)
        print("Macierz transformacji:")
        print(reg_cicp.transformation)
        
        information_reg_cicp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, threshold, reg_cicp.transformation
        )
        return reg_cicp.transformation, information_reg_cicp
    
    else:
        print('Nie wybrano odpowiedniego sposobu transformacji')
        return None, None


chmura_tls = o3d.io.read_point_cloud("woksele.pcd")
chmura_rgb = o3d.io.read_point_cloud("odfiltrowanee57.pcd")

trans_init = np.identity(4)
source_centroid = np.mean(chmura_tls.points, axis=0)
target_centroid = np.mean(chmura_rgb.points, axis=0)
trans_init[:3, 3] = target_centroid - source_centroid
metoda = 'p2p'

transf, info = ICP_registration(chmura_tls, chmura_rgb, threshold=0.1, metoda=metoda, trans_init=trans_init, max_iteration=1000)
print(transf)
print(info)
# save translated point cloud
o3d.io.write_point_cloud(f"woksele_transformed_{metoda}.pcd", chmura_tls.transform(transf))

np.set_printoptions(suppress=True)
# save information matrix
np.savetxt(f"info_{metoda}.txt", info)

# save transformation matrix
np.savetxt(f"transf_{metoda}.txt", transf)

