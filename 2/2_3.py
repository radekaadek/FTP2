import click
import open3d as o3d
import numpy as np

@click.command()
@click.argument(
    'input_pcd',
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    '--normal-radius',
    type=click.FLOAT,
    default=0.1,
    show_default=True,
    help='Radius for KDTree search during normal estimation.'
)
@click.option(
    '--normal-max-nn',
    type=click.INT,
    default=30,
    show_default=True,
    help='Maximum neighbors for KDTree search during normal estimation.'
)
@click.option(
    '--poisson-depth',
    type=click.INT,
    default=9,
    show_default=True,
    help='Depth parameter for Poisson reconstruction (used only if --poisson-output is set).'
)
@click.option(
    '--crop/--no-crop',
    default=True,
    show_default=True,
    help='Crop the Poisson mesh to the bounding box of the input cloud (used only if --poisson-output is set).'
)
@click.option(
    '--poisson-output',
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help='If specified, perform Poisson reconstruction and save the resulting mesh to this file path (e.g., poisson_mesh.ply).'
)
@click.option(
    '--ball-pivoting-output',
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help='If specified, perform Ball Pivoting reconstruction and save the resulting mesh to this file path (e.g., bp_mesh.stl).'
)
def process_cloud(input_pcd, normal_radius, normal_max_nn,
                  poisson_depth, crop, poisson_output,
                  ball_pivoting_output):
    """
    Processes a point cloud file (INPUT_PCD) using Open3D.

    Estimates normals and optionally reconstructs the surface using
    Poisson Surface Reconstruction and/or Ball Pivoting algorithms,
    saving the results if output paths are provided.
    """
    bp_radii_mult = [1.5, 2.0, 2.5, 3.0] # Multipliers for average distance

    if not poisson_output and not ball_pivoting_output:
        click.echo("Error: No output file specified for Poisson (--poisson-output) "
                   "or Ball Pivoting (--ball-pivoting-output). "
                   "The script will only load the cloud and estimate normals. "
                   "Provide an output path to run reconstruction.", err=True)
        return 1

    click.echo(f"Loading point cloud from: {input_pcd}")
    try:
        cloud = o3d.io.read_point_cloud(input_pcd)
        if not cloud.has_points():
            click.echo(f"Error: Point cloud file {input_pcd} is empty or could not be read correctly.", err=True)
            return 1
    except Exception as e:
        click.echo(f"Error loading point cloud: {e}", err=True)
        return 1

    click.echo("Estimating normals...")
    try:
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
        cloud.normalize_normals()
        click.echo(f"Normals estimated (radius={normal_radius}, max_nn={normal_max_nn}) and normalized.")
    except Exception as e:
        click.echo(f"Error estimating normals: {e}", err=True)

    if not cloud.has_normals():
         click.echo("Warning: Normals could not be estimated successfully. Reconstruction might fail or produce poor results.", err=True)

    if poisson_output:
        if not cloud.has_normals():
             click.echo("Skipping Poisson reconstruction: Input cloud has no normals.", err=True)
        else:
            click.echo(f"Performing Poisson reconstruction (depth={poisson_depth})...")
            try:
                poisson_mesh_result, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=poisson_depth)
                if not poisson_mesh_result or not poisson_mesh_result.has_triangles():
                    click.echo("Warning: Poisson reconstruction resulted in an empty or invalid mesh. No file saved.", err=True)
                else:
                    click.echo("Poisson reconstruction successful.")
                    if crop:
                        click.echo("Cropping Poisson mesh to input cloud bounds...")
                        bbox = cloud.get_axis_aligned_bounding_box()
                        poisson_mesh_result_cropped = poisson_mesh_result.crop(bbox)
                        if not poisson_mesh_result_cropped.has_triangles():
                             click.echo("Warning: Cropping removed all triangles from the Poisson mesh. Saving original.", err=True)
                        else:
                             poisson_mesh_result = poisson_mesh_result_cropped
                             click.echo("Cropping complete.")


                    click.echo(f"Saving Poisson mesh to: {poisson_output}")
                    if not o3d.io.write_triangle_mesh(poisson_output, poisson_mesh_result, write_ascii=False, compressed=True):
                        click.echo(f"Error: Failed to save Poisson mesh to {poisson_output}", err=True)
                    else:
                        click.echo("Poisson mesh saved.")

            except Exception as e:
                click.echo(f"Error during Poisson reconstruction: {e}", err=True)

    if ball_pivoting_output:
        if not cloud.has_normals():
             click.echo("Skipping Ball Pivoting reconstruction: Input cloud has no normals.", err=True)
        else:
            click.echo("Calculating average point distance for Ball Pivoting...")
            try:
                dists = cloud.compute_nearest_neighbor_distance()
                avg_dist = np.mean(dists)
                if avg_dist <= 0:
                     click.echo("Error: Could not compute valid average point distance for Ball Pivoting.", err=True)
                else:
                    click.echo(f"Average distance between points: {avg_dist:.4f}")

                    radii_values = [avg_dist * m for m in bp_radii_mult]
                    if not radii_values:
                        click.echo("Error: No radii specified or calculated for Ball Pivoting.", err=True)
                    else:
                        click.echo(f"Using Ball Pivoting radii: {radii_values}")
                        ball_pivoting_mesh_result = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                            cloud,
                            o3d.utility.DoubleVector(radii_values)
                        )

                        if not ball_pivoting_mesh_result or not ball_pivoting_mesh_result.has_triangles():
                            click.echo("Warning: Ball Pivoting reconstruction resulted in an empty or invalid mesh. No file saved.", err=True)
                        else:
                            click.echo("Ball Pivoting reconstruction successful.")
                            click.echo(f"Saving Ball Pivoting mesh to: {ball_pivoting_output}")
                            if not o3d.io.write_triangle_mesh(ball_pivoting_output, ball_pivoting_mesh_result, write_ascii=False, compressed=True):
                                click.echo(f"Error: Failed to save Ball Pivoting mesh to {ball_pivoting_output}", err=True)
                            else:
                                click.echo("Ball Pivoting mesh saved.")

            except Exception as e:
                click.echo(f"Error during Ball Pivoting setup or reconstruction: {e}", err=True)

    click.echo("Processing finished.")
    return 0


if __name__ == "__main__":
    process_cloud()
