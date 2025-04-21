import open3d as o3d
import numpy as np
import click
import os

def icp_registration(source, target, threshold, trans_init, method, max_iteration):
    """
    Performs ICP registration between two point clouds.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud (to be transformed).
        target (o3d.geometry.PointCloud): The target point cloud.
        threshold (float): Maximum correspondence distance for ICP.
        trans_init (np.ndarray): Initial 4x4 transformation matrix.
        method (str): ICP method ('p2p', 'p2pl', 'cicp').
        max_iteration (int): Maximum number of ICP iterations.

    Returns:
        tuple: (transformation_matrix, information_matrix) or (None, None) on failure.
    """
    click.echo('\n--- Analiza dokładności wstępnej orientacji ---')
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    click.echo(f"Fitness: {evaluation.fitness:.4f}")
    click.echo(f"Inlier RMSE: {evaluation.inlier_rmse:.4f}")
    click.echo(f"Correspondence set size: {len(evaluation.correspondence_set)}")

    conv_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iteration)

    reg_result = None # To store the registration result object

    if method == 'p2p':
        click.echo("\n--- Orientacja ICP <Punkt do punktu> ---")
        reg_result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=conv_criteria
        )

    elif method == 'p2pl':
        click.echo('\n--- Wyznaczanie normalnych dla metody Punkt do Płaszczyzny ---')
        if not source.has_normals():
             click.echo("Wyznaczanie normalnych dla chmury źródłowej...")
             source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
        if not target.has_normals():
            click.echo("Wyznaczanie normalnych dla chmury docelowej...")
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
        click.echo("\n--- Orientacja ICP <Punkt do płaszczyzny> ---")
        reg_result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=conv_criteria
        )

    elif method == 'cicp':
        click.echo("\n--- Orientacja ICP <Kolorowy ICP> ---")
        if not source.has_colors():
            click.echo(click.style("Ostrzeżenie: Chmura źródłowa nie ma kolorów dla C-ICP.", fg='yellow'))
            # Assign dummy colors if needed, or handle error
            # source.paint_uniform_color([0.5, 0.5, 0.5])
        if not target.has_colors():
             click.echo(click.style("Ostrzeżenie: Chmura docelowa nie ma kolorów dla C-ICP.", fg='yellow'))
             # target.paint_uniform_color([0.5, 0.5, 0.5])

        try:
            reg_result = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                criteria=conv_criteria
            )
        except Exception as e:
            click.echo(click.style(f"Błąd podczas Colored ICP: {e}", fg='red'))
            return None, None


    else:
        click.echo(click.style(f"Nieznana metoda ICP: {method}", fg='red'))
        return None, None

    if reg_result:
        click.echo("\n--- Wyniki Rejestracji ---")
        click.echo(f"Fitness: {reg_result.fitness:.4f}")
        click.echo(f"Inlier RMSE: {reg_result.inlier_rmse:.4f}")
        click.echo(f"Correspondence set size: {len(reg_result.correspondence_set)}")
        click.echo("\nMacierz transformacji:")
        click.echo(reg_result.transformation)

        click.echo("\n--- Obliczanie Macierzy Informacji ---")
        try:
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source, target, threshold, reg_result.transformation
            )
            return reg_result.transformation, information_matrix
        except Exception as e:
            click.echo(click.style(f"Nie można obliczyć macierzy informacji: {e}", fg='red'))
            return reg_result.transformation, None
    else:
         click.echo(click.style("Rejestracja ICP nie powiodła się.", fg='red'))
         return None, None


@click.command()
@click.argument('source_file', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('target_file', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--output-prefix', '-o', default=None, type=str,
              help='Prefix for output files (e.g., "reg_result_"). If not provided, uses the source filename base.')
@click.option('--method', '-m', default='p2p',
              type=click.Choice(['p2p', 'p2pl', 'cicp'], case_sensitive=False),
              help='ICP registration method.')
@click.option('--threshold', '-t', default=0.1, type=click.FLOAT,
              help='ICP correspondence distance threshold.')
@click.option('--max-iterations', '-i', default=1000, type=click.INT,
              help='Maximum number of ICP iterations.')
@click.option('--initial-transform-centroid', is_flag=True, default=True,
              help='Use centroid difference for initial translation (default). Use --no-initial-transform-centroid for identity matrix.')
@click.option('--no-initial-transform-centroid', is_flag=True, default=False,
              help='Do not use centroid difference; use identity matrix for initial transform.')
@click.option('--save-transformed-cloud', is_flag=True, default=True, help='Save the transformed source point cloud.')
@click.option('--no-save-transformed-cloud', is_flag=True, default=False, help='Do not save the transformed source point cloud.')

def main(source_file, target_file, output_prefix, method, threshold, max_iterations, initial_transform_centroid, no_initial_transform_centroid, save_transformed_cloud, no_save_transformed_cloud):
    """
    Rejestruje chmurę punktów SOURCE_FILE do TARGET_FILE za pomocą algorytmu ICP.

    Zapisuje wynikową macierz transformacji, macierz informacji i opcjonalnie
    przetransformowaną chmurę źródłową.
    """
    click.echo(f"Ładowanie chmury źródłowej: {source_file}")
    try:
        chmura_source = o3d.io.read_point_cloud(source_file)
        if not chmura_source.has_points():
            click.echo(click.style(f"Błąd: Plik źródłowy {source_file} nie zawiera punktów.", fg='red'))
            return 1
    except Exception as e:
        click.echo(click.style(f"Błąd podczas ładowania pliku źródłowego {source_file}: {e}", fg='red'))
        return 1

    click.echo(f"Ładowanie chmury docelowej: {target_file}")
    try:
        chmura_target = o3d.io.read_point_cloud(target_file)
        if not chmura_target.has_points():
            click.echo(click.style(f"Błąd: Plik docelowy {target_file} nie zawiera punktów.", fg='red'))
            return 1
    except Exception as e:
        click.echo(click.style(f"Błąd podczas ładowania pliku docelowego {target_file}: {e}", fg='red'))
        return 1

    trans_init = np.identity(4)
    use_centroid_init = initial_transform_centroid and not no_initial_transform_centroid

    if use_centroid_init:
        click.echo("Obliczanie wstępnej transformacji na podstawie centroidów...")
        try:
            source_centroid = chmura_source.get_center() # More robust than np.mean
            target_centroid = chmura_target.get_center()
            trans_init[:3, 3] = target_centroid - source_centroid
            click.echo(f"  Wstępne przesunięcie: {trans_init[:3, 3]}")
        except Exception as e:
             click.echo(click.style(f"Nie można obliczyć centroidów, używanie macierzy jednostkowej: {e}", fg='yellow'))
             trans_init = np.identity(4)
    else:
        click.echo("Używanie macierzy jednostkowej jako wstępnej transformacji.")
        trans_init = np.identity(4)

    click.echo(f"\nRozpoczynanie rejestracji ICP metodą: {method}")
    click.echo(f"Próg: {threshold}, Maks. iteracji: {max_iterations}")

    transformation, information = icp_registration(
        chmura_source, chmura_target,
        threshold=threshold,
        trans_init=trans_init,
        method=method,
        max_iteration=max_iterations
    )

    if transformation is not None:
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            output_prefix = f"{base_name}_registered_"
        else:
            if not output_prefix.endswith(('_', '-', '.')):
                 output_prefix += "_"


        transform_file = f"{output_prefix}transf_{method}.txt"
        info_file = f"{output_prefix}info_{method}.txt"
        transformed_cloud_file = f"{output_prefix}transformed_{method}.pcd"

        click.echo(f"\nZapisywanie macierzy transformacji do: {transform_file}")
        try:
            np.savetxt(transform_file, transformation, fmt='%.8f')
        except Exception as e:
             click.echo(click.style(f"Nie można zapisać macierzy transformacji: {e}", fg='red'))

        if information is not None:
            click.echo(f"Zapisywanie macierzy informacji do: {info_file}")
            try:
                np.savetxt(info_file, information, fmt='%.8f')
            except Exception as e:
                 click.echo(click.style(f"Nie można zapisać macierzy informacji: {e}", fg='red'))
        else:
             click.echo(click.style("Macierz informacji nie została obliczona lub zapisana.", fg='yellow'))


        should_save_cloud = save_transformed_cloud and not no_save_transformed_cloud
        if should_save_cloud:
             click.echo(f"Transformowanie i zapisywanie chmury źródłowej do: {transformed_cloud_file}")
             try:
                 # Apply the final transformation IN PLACE for saving
                 chmura_source_transformed = chmura_source.transform(transformation)
                 o3d.io.write_point_cloud(transformed_cloud_file, chmura_source_transformed)
             except Exception as e:
                 click.echo(click.style(f"Nie można zapisać przetransformowanej chmury punktów: {e}", fg='red'))
        else:
            click.echo("Pomijanie zapisywania przetransformowanej chmury źródłowej.")

        click.echo(click.style("\nRejestracja zakończona pomyślnie.", fg='green'))
        return 0
    else:
        click.echo(click.style("\nRejestracja nie powiodła się.", fg='red'))
        return 1

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=8)
    main()
