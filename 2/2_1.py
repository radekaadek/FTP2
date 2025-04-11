import laspy
import open3d as o3d
import numpy as np
import os
import click
from pathlib import Path

def wczytanie_chmury_punktow_las_konwersja_do_o3d(plik: Path) -> o3d.geometry.PointCloud:
    """
    Wczytuje chmurę punktów z pliku LAS/LAZ i konwertuje ją do formatu Open3D.
    (Implementacja jak w poprzedniej odpowiedzi)
    """
    if not plik.is_file():
        raise FileNotFoundError(f"Plik nie został znaleziony: {plik}")

    try:
        las_pcd = laspy.read(str(plik))
        points = np.vstack((las_pcd.x, las_pcd.y, las_pcd.z)).transpose()
        chmura_punktow = o3d.geometry.PointCloud()
        chmura_punktow.points = o3d.utility.Vector3dVector(points)

        if hasattr(las_pcd, 'red') and hasattr(las_pcd, 'green') and hasattr(las_pcd, 'blue'):
            max_val = np.iinfo(las_pcd.red.dtype).max if np.issubdtype(las_pcd.red.dtype, np.integer) else 1.0
            if max_val == 0: max_val = 1.0
            if np.max(las_pcd.red) <= 1.0 and np.max(las_pcd.green) <= 1.0 and np.max(las_pcd.blue) <= 1.0:
                 r, g, b = las_pcd.red, las_pcd.green, las_pcd.blue
            else:
                 r = las_pcd.red / max_val
                 g = las_pcd.green / max_val
                 b = las_pcd.blue / max_val

            colors = np.vstack((r, g, b)).transpose()
            colors = np.clip(colors, 0.0, 1.0)
            chmura_punktow.colors = o3d.utility.Vector3dVector(colors)
        else:
            click.echo(f"Ostrzeżenie: Brak informacji o kolorze w pliku {plik.name}. Ta część chmury będzie bez kolorów.")

        return chmura_punktow

    except Exception as e:
        raise Exception(f"Błąd podczas wczytywania pliku {plik}: {e}")


def wyznaczanie_obserwacji_odstajacych(chmura_punktow: o3d.geometry.PointCloud,
                                        liczba_sasiadow: int = 30,
                                        std_ratio: float = 2.0) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Usuwa punkty odstające z chmury punktów metodą statystyczną.
    (Implementacja jak w poprzedniej odpowiedzi)
    """
    if not chmura_punktow.has_points():
        click.echo("Ostrzeżenie: Próba usunięcia punktów odstających z pustej chmury.")
        return chmura_punktow, o3d.geometry.PointCloud()

    chmura_punktow_odfiltrowana, ind = chmura_punktow.remove_statistical_outlier(
        nb_neighbors=liczba_sasiadow, std_ratio=std_ratio
    )
    punkty_odstajace = chmura_punktow.select_by_index(ind, invert=True)
    return chmura_punktow_odfiltrowana, punkty_odstajace

def regularyzacja_chmury_punktow(chmura_punktow: o3d.geometry.PointCloud,
                                 rozmiar_woksela: float = 0.1) -> o3d.geometry.PointCloud:
    """
    Downsampling chmury punktów metodą siatki wokseli.
    (Implementacja jak w poprzedniej odpowiedzi)
    """
    if not chmura_punktow.has_points():
       click.echo("Ostrzeżenie: Próba wokselizacji pustej chmury.")
       return chmura_punktow

    chmura_punktow_woksele = chmura_punktow.voxel_down_sample(voxel_size=rozmiar_woksela)
    return chmura_punktow_woksele

def usuwanie_co_n_tego_punktu_z_chmury_punktow(chmura_punktow: o3d.geometry.PointCloud,
                                                co_n_ty_punkt: int = 2) -> o3d.geometry.PointCloud:
    """
    Downsampling chmury punktów przez zachowanie co n-tego punktu.
    (Implementacja jak w poprzedniej odpowiedzi)
    """
    if not chmura_punktow.has_points():
      click.echo("Ostrzeżenie: Próba jednolitego downsamplingu pustej chmury.")
      return chmura_punktow
    if co_n_ty_punkt <= 0:
        click.echo("Ostrzeżenie: 'co_n_ty_punkt' musi być większe od 0. Zwracam oryginalną chmurę.")
        return chmura_punktow

    chmura_punktow_co_n_ty = chmura_punktow.uniform_down_sample(every_k_points=co_n_ty_punkt)
    return chmura_punktow_co_n_ty

@click.command()
@click.argument('input_path', type=click.Path(exists=True, readable=True, path_type=Path))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False, writable=True, path_type=Path), default='output_combined',
              help='Katalog wyjściowy dla połączonego pliku (domyślnie: ./output_combined).')
@click.option('--output-basename', type=str, default='combined',
              help='Podstawowa nazwa dla plików wyjściowych (domyślnie: combined).')
@click.option('--remove-outliers', is_flag=True, help='Włącz usuwanie punktów odstających z połączonej chmury.')
@click.option('--outlier-neighbors', type=int, default=30, help='Liczba sąsiadów dla usuwania odstających.')
@click.option('--outlier-std-ratio', type=float, default=2.0, help='Współczynnik odch. std. dla usuwania odstających.')
@click.option('--uniform-downsample', is_flag=True, help='Włącz jednolity downsampling (co n-ty punkt) połączonej chmury.')
@click.option('--uniform-k', type=int, default=2, help='Zachowaj co k-ty punkt podczas jednolitego downsamplingu.')
@click.option('--voxel-downsample', is_flag=True, help='Włącz downsampling wokselowy połączonej chmury.')
@click.option('--voxel-size', type=float, default=0.1, help='Rozmiar woksela dla downsamplingu.')
@click.option('--save-outliers', is_flag=True, help='Zapisz punkty odstające (z połączonej chmury) do osobnego pliku.')
@click.option('--save-voxels', is_flag=True, help='Zapisz zwokselizowaną połączoną chmurę do osobnego pliku.')
@click.option('--output-format', type=click.Choice(['pcd', 'ply']), default='pcd', help='Format zapisu plików wyjściowych.')
def process_point_clouds_combined(input_path: Path, output_dir: Path, output_basename: str,
                                   remove_outliers: bool, outlier_neighbors: int, outlier_std_ratio: float,
                                   uniform_downsample: bool, uniform_k: int,
                                   voxel_downsample: bool, voxel_size: float,
                                   save_outliers: bool, save_voxels: bool, output_format: str):
    """
    Łączy chmury punktów z pliku LAS/LAZ lub katalogu w jedną, przetwarza ją
    i zapisuje jako pojedynczy plik.

    INPUT_PATH: Ścieżka do pliku .las/.laz lub katalogu zawierającego takie pliki.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Katalog wyjściowy: {output_dir.resolve()}")

    files_to_process = []
    if input_path.is_file():
        if input_path.suffix.lower() in ['.las', '.laz']:
            files_to_process.append(input_path)
        else:
            click.echo(f"Błąd: Oczekiwano pliku .las lub .laz, otrzymano: {input_path}", err=True)
            return 1
    elif input_path.is_dir():
        click.echo(f"Przeszukiwanie katalogu: {input_path.resolve()}")
        for ext in ['*.las', '*.laz']:
            files_to_process.extend(input_path.glob(ext))
        if not files_to_process:
            click.echo(f"Ostrzeżenie: Nie znaleziono plików .las ani .laz w katalogu: {input_path}", err=True)
            return 0
    else:
        click.echo(f"Błąd: Nieprawidłowa ścieżka wejściowa: {input_path}", err=True)
        return 1

    if not files_to_process:
        click.echo("Nie znaleziono plików do przetworzenia.")
        return 0

    click.echo(f"Znaleziono {len(files_to_process)} plików do połączenia i przetworzenia.")
    click.echo("Ostrzeżenie: Łączenie wielu dużych plików może wymagać znacznej ilości pamięci RAM.")

    combined_cloud = o3d.geometry.PointCloud()
    has_color = False

    click.echo("\n--- Łączenie chmur punktów ---")
    for i, file_path in enumerate(files_to_process):
        click.echo(f"[{i+1}/{len(files_to_process)}] Wczytywanie: {file_path.name} ...", nl=False)
        try:
            chmura = wczytanie_chmury_punktow_las_konwersja_do_o3d(file_path)
            if chmura.has_points():
                combined_cloud += chmura
                if chmura.has_colors():
                    has_color = True
                click.echo(f" OK ({len(chmura.points)} pkt)")
            else:
                 click.echo(" Pusty plik, pomijanie.")

        except Exception as e:
            click.echo(f"\nBłąd podczas wczytywania pliku {file_path.name}: {e}", err=True)
            click.echo("Kontynuowanie z następnym plikiem...")

    if not combined_cloud.has_points():
        click.echo("\nBrak punktów w połączonej chmurze. Zakończono.", err=True)
        return 0

    click.echo(f"\n--- Połączono chmurę: {len(combined_cloud.points)} punktów ---")
    if not has_color and combined_cloud.has_colors():
         combined_cloud.colors = o3d.utility.Vector3dVector()
         click.echo("Usunięto domyślne kolory, ponieważ żaden plik wejściowy ich nie zawierał.")
    elif has_color:
         click.echo("Połączona chmura zawiera informacje o kolorze.")


    click.echo("\n--- Przetwarzanie połączonej chmury ---")
    processed_cloud = combined_cloud
    outlier_cloud = None
    voxel_cloud = None

    if uniform_downsample:
        click.echo(f"Stosowanie jednolitego downsamplingu (co {uniform_k}-ty punkt)...")
        processed_cloud = usuwanie_co_n_tego_punktu_z_chmury_punktow(processed_cloud, uniform_k)
        click.echo(f" -> Pozostało punktów: {len(processed_cloud.points)}")

    if remove_outliers:
        click.echo(f"Usuwanie punktów odstających (sąsiedzi={outlier_neighbors}, std_ratio={outlier_std_ratio})...")
        processed_cloud, outlier_cloud = wyznaczanie_obserwacji_odstajacych(
            processed_cloud, outlier_neighbors, outlier_std_ratio
        )
        click.echo(f" -> Pozostało punktów: {len(processed_cloud.points)}")
        if outlier_cloud and outlier_cloud.has_points():
            click.echo(f" -> Znaleziono punktów odstających: {len(outlier_cloud.points)}")
        else:
            click.echo(" -> Nie znaleziono punktów odstających.")

    if voxel_downsample:
        click.echo(f"Stosowanie downsamplingu wokselowego (rozmiar={voxel_size})...")
        voxel_result = regularyzacja_chmury_punktow(processed_cloud, voxel_size)
        if save_voxels:
            voxel_cloud = voxel_result
            click.echo(f" -> Punktów po wokselizacji (do zapisu): {len(voxel_cloud.points)}")
        else:
            processed_cloud = voxel_result
            click.echo(f" -> Pozostało punktów po wokselizacji: {len(processed_cloud.points)}")


    click.echo("\n--- Zapisywanie wyników ---")

    output_processed_path = output_dir / f"{output_basename}_processed.{output_format}"
    click.echo(f"Zapisywanie połączonej i przetworzonej chmury do: {output_processed_path}")
    if not o3d.io.write_point_cloud(str(output_processed_path), processed_cloud, write_ascii=False, compressed=True):
         click.echo(f"Błąd: Nie udało się zapisać pliku {output_processed_path}", err=True)
         return 1

    if remove_outliers and save_outliers and outlier_cloud and outlier_cloud.has_points():
        output_outliers_path = output_dir / f"{output_basename}_outliers.{output_format}"
        click.echo(f"Zapisywanie punktów odstających do: {output_outliers_path}")
        if not o3d.io.write_point_cloud(str(output_outliers_path), outlier_cloud, write_ascii=False, compressed=True):
             click.echo(f"Błąd: Nie udało się zapisać pliku {output_outliers_path}", err=True)

    if voxel_downsample and save_voxels and voxel_cloud and voxel_cloud.has_points():
         output_voxels_path = output_dir / f"{output_basename}_voxels.{output_format}"
         click.echo(f"Zapisywanie zwokselizowanej chmury do: {output_voxels_path}")
         if not o3d.io.write_point_cloud(str(output_voxels_path), voxel_cloud, write_ascii=False, compressed=True):
              click.echo(f"Błąd: Nie udało się zapisać pliku {output_voxels_path}", err=True)

    click.echo("\n--- Zakończono pomyślnie ---")
    return 0

if __name__ == '__main__':
    process_point_clouds_combined()
