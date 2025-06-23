from ftplib import FTP_TLS
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import subprocess
import os
import shutil

seven_zip_exe = r"D:\7-Zip\7z.exe"

def download_doris_rinex(
    start_datetime: datetime,
    days: int,
    satellite: str,
    base_dir: str = "./DORISInput/rinexobs"
):

    date_list = [start_datetime + timedelta(days=i) for i in range(days)]

    doy_map = {}
    for dt in date_list:
        year = dt.year
        yy = str(year)[2:]
        doy = f"{dt.timetuple().tm_yday:03}"
        prefix = f"{satellite}rx{yy}{doy}"
        doy_map.setdefault(year, []).append(prefix)

    ftp_host = "gdc.cddis.eosdis.nasa.gov"
    ftp_base_path = "/doris/data"

    with tqdm(total=len(date_list), desc=f"Downloading {satellite}") as pbar:
        for year, prefixes in doy_map.items():
            ftp_path = f"{ftp_base_path}/{satellite}/{year}/"
            local_dir = os.path.join(base_dir, satellite, str(year))
            os.makedirs(local_dir, exist_ok=True)

            try:
                ftps = FTP_TLS(ftp_host)
                ftps.login()
                ftps.prot_p()
                ftps.cwd(ftp_path)
                files = ftps.nlst()
            except Exception as e:
                print(f"Cannot access {ftp_path}: {e}")
                for _ in prefixes:
                    pbar.update(1)
                continue

            for prefix in prefixes:
                matched_files = [f for f in files if f.startswith(prefix) and f.endswith(".Z")]
                if not matched_files:
                    print(f"No data for {prefix}")
                    pbar.update(1)
                    continue

                for fname in matched_files: # actually only one matched file will exist normally
                    local_path = os.path.join(local_dir, fname)
                    
                    if os.path.exists(local_path.replace(".Z", "")):
                        pbar.update(1)
                        continue
                    try:
                        with open(local_path, "wb") as f:
                            ftps.retrbinary(f"RETR {fname}", f.write)

                        if local_path.endswith(".Z"):
                            subprocess.run([seven_zip_exe, "x", str(local_path), f"-o{str(local_dir)}", "-y"],                     
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=True)               
                           
                        else:
                            shutil.unpack_archive(local_path, extract_dir=local_dir)

                        os.remove(local_path)
                    except Exception as e:
                        print(f"Failed to download/decompress {fname}: {e}")
                pbar.update(1)

            ftps.quit()

def download_doris_sp3(
    start_datetime: datetime,
    days: int,
    satellite: str,
    base_dir: str = "./DORISInput/sp3"
):

    ftp_host = "gdc.cddis.eosdis.nasa.gov"
    ftp_base_path = f"/doris/products/orbits/ssa/{satellite}/"
    os.makedirs(os.path.join(base_dir, satellite), exist_ok=True)

    date_list = [start_datetime + timedelta(days=i) for i in range(days)]

    target_doys = set(f"{dt.year % 100:02}{dt.timetuple().tm_yday:03}" for dt in date_list)

    try:
        ftps = FTP_TLS(ftp_host)
        ftps.login()
        ftps.prot_p()
        ftps.cwd(ftp_base_path)
        filenames = ftps.nlst()
    except Exception as e:
        print(f"FTP connection failed: {e}")
        return

    matched_files = []
    for fname in filenames:

        if not fname.endswith(".sp3.001.Z"):
            continue
        for doy in target_doys:
            if f"b{doy}" in fname or f"e{doy}" in fname:
                matched_files.append(fname)
                break

    if not matched_files:
        print("No matching SP3 files found for the given date range.")
        return

    with tqdm(total=len(matched_files), desc=f"Downloading SP3 for {satellite}") as pbar:
        for fname in matched_files:
            local_path = os.path.join(base_dir, satellite, fname)
            try:
                with open(local_path, "wb") as f:
                    ftps.retrbinary(f"RETR {fname}", f.write)

                if local_path.endswith(".Z"):
                    subprocess.run([seven_zip_exe, "x", str(local_path), f"-o{str(os.path.join(base_dir, satellite))}", "-y"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True)               
                    
                else:
                    shutil.unpack_archive(local_path, os.path.join(base_dir, satellite))
                os.remove(local_path)
            except Exception as e:
                print(f"Failed to download or decompress {fname}: {e}")
            pbar.update(1)

    ftps.quit()

def download_gim(
    start_datetime: datetime,
    days: int,
    base_dir: str = "./DORISInput/IGSGIM"
):
    ftp_host = "gdc.cddis.eosdis.nasa.gov"
    ftp_base_path = "/gnss/products/ionex"
    os.makedirs(base_dir, exist_ok=True)

    # Generate list of date objects
    date_list = [start_datetime + timedelta(days=i) for i in range(days)]

    with tqdm(total=len(date_list), desc="Downloading GIM") as pbar:
        for dt in date_list:
            year = dt.year
            yy = str(year % 100).zfill(2)
            doy = dt.timetuple().tm_yday
            doy_str = f"{doy:03}"

            subdir = f"{ftp_base_path}/{year}/{doy_str}/"
            local_dir = os.path.join(base_dir, str(year))
            os.makedirs(local_dir, exist_ok=True)

            try:
                ftps = FTP_TLS(ftp_host)
                ftps.login()
                ftps.prot_p()
                ftps.cwd(subdir)
                filenames = ftps.nlst()
            except Exception as e:
                print(f"Cannot access {subdir}: {e}")
                pbar.update(1)
                continue

            # Select preferred file
            inx_file = next((f for f in filenames if f.startswith(f"IGS0OPSFIN_{year}{doy_str}") and f.endswith("GIM.INX.gz")), None)
            igsg_file = next((f for f in filenames if f.startswith(f"igsg{doy_str}0.{yy}i.Z")), None)

            selected_file = inx_file or igsg_file

            if not selected_file:
                print(f"No GIM file found for {dt.strftime('%Y-%m-%d')}")
                pbar.update(1)
                continue

            local_path = os.path.join(local_dir, selected_file)
            try:
                with open(local_path, "wb") as f:
                    ftps.retrbinary(f"RETR {selected_file}", f.write)

                if local_path.endswith(('.Z', '.gz')):
                    subprocess.run([seven_zip_exe, "x", str(local_path), f"-o{str(local_dir)}", "-y"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True)                 
                    
                else:
                    shutil.unpack_archive(local_path, extract_dir=local_dir)

                os.remove(local_path)
            except Exception as e:
                print(f"Failed to download or decompress {selected_file}: {e}")

            ftps.quit()
            pbar.update(1)

    print("GIM download complete.")

if __name__ == '__main__':
    start_time = datetime(2019, 11, 30)
    days = 120
    satellite = 'ja3'

    # download_doris_rinex(start_time, days, satellite)
    # download_doris_sp3(start_time, days, satellite)
    download_gim(start_time, days)