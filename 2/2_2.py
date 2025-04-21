import open3d as o3d
import numpy as np
import click

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

    # Move both clouds to 0,0,0 remember the target previous position
    source.translate(-source.get_center())
    target_translation = target.get_center()
    target.translate(-target_translation)


    click.echo('\n--- Analiza dokładności wstępnej orientacji ---')
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    click.echo(f"Fitness: {evaluation.fitness}")
    click.echo(f"Inlier RMSE: {evaluation.inlier_rmse:}")
    click.echo(f"Rozmiar correspondence set: {len(evaluation.correspondence_set)}")

    conv_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iteration)

    reg_result = None

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
        if not target.has_normals():
             click.echo(click.style("Ostrzeżenie: Chmura docelowa nie ma normalów dla C-ICP. Wyznaczanie wektorów normalnych...", fg='yellow'))
             target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))

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

    transformation_matrix = reg_result.transformation.copy()
    transformation_matrix[:3, 3] += target_translation

    if reg_result:
        click.echo("\n--- Wyniki Rejestracji ---")
        click.echo(f"Fitness: {reg_result.fitness:.4f}")
        click.echo(f"Inlier RMSE: {reg_result.inlier_rmse:.4f}")
        click.echo(f"Rozmiar correspondence set: {len(reg_result.correspondence_set)}")
        click.echo("\nMacierz transformacji:")
        click.echo(transformation_matrix)

        click.echo("\n--- Obliczanie Macierzy Informacji ---")
        try:
            source_copy_for_info = o3d.geometry.PointCloud(source)
            source_copy_for_info.transform(reg_result.transformation)
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_copy_for_info, target, threshold, np.identity(4)
            )

            return transformation_matrix, information_matrix
        except Exception as e:
            click.echo(click.style(f"Nie można obliczyć macierzy informacji: {e}", fg='red'))
            return transformation_matrix, None
    else:
         click.echo(click.style("Rejestracja ICP nie powiodła się.", fg='red'))
         return None, None


@click.command()
@click.argument('source_file', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('target_file', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--output-cloud', '-oc', default=None,
              type=click.Path(dir_okay=False, writable=True),
              help='Output file for the transformed source point cloud (e.g., transformed.pcd).')
@click.option('--method', '-m', default='p2p',
              type=click.Choice(['p2p', 'p2pl', 'cicp'], case_sensitive=False),
              help='ICP registration method.')
@click.option('--threshold', '-t', default=0.1, type=click.FLOAT,
              help='ICP correspondence distance threshold.')
@click.option('--max-iterations', '-i', default=1000, type=click.INT,
              help='Maximum number of ICP iterations.')
def main(source_file, target_file, output_cloud, method, threshold, max_iterations):
    """
    Rejestruje chmurę punktów SOURCE_FILE do TARGET_FILE za pomocą algorytmu ICP.

    Zapisuje opcjonalnie wynikową macierz transformacji (--output-transform),
    macierz informacji (--output-info) i przetransformowaną chmurę źródłową (--output-cloud)
    do podanych plików.
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
        if output_cloud:
            click.echo(f"Transformacja i zapisywanie chmury źródłowej do: {output_cloud}")
            try:
                # Create a copy of the *original* source cloud and apply the *final* transformation for saving
                chmura_source_copy_to_save = o3d.geometry.PointCloud(chmura_source) # Make a copy from potentially modified source
                chmura_source_transformed = chmura_source_copy_to_save.transform(transformation)
                o3d.io.write_point_cloud(output_cloud, chmura_source_transformed)
            except Exception as e:
                click.echo(click.style(f"Nie można zapisać przetransformowanej chmury punktów do {output_cloud}: {e}", fg='red'))
        click.echo(click.style("\nRejestracja zakończona pomyślnie.", fg='green'))
        return 0
    else:
        click.echo(click.style("\nRejestracja nie powiodła się.", fg='red'))
        return 1

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=8)
    main()
