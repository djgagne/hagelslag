from datetime import datetime, timedelta 
from os.path import exists
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", required=True, help="Start date for symbolic links in YYYY-MM-DD format")
    parser.add_argument("-e", "--end_date", required=False, help="End date for symbolic links in YYYY-MM-DD format")
    parser.add_argument("-p", "--source_path", required=True, help="Path to the source directory of HREFv2 dataset")
    parser.add_argument("-o", "--destination_path", required=True, help="Path to the destination of symbolic HREFv2 dataset")
    args = parser.parse_args()
    source_path = args.source_path
    destination_path = args.destination_path 
    if args.end_date:
        date_index = pd.DatetimeIndex(start=args.start_date, end=args.end_date, freq='1D')
    else:
        date_index = pd.DatetimeIndex(start=args.start_date, end=args.start_date, freq='1D')

    member = ['nam_00' ,'nam_12','nmmb_00','nmmb_12','nssl_00','nssl_12','arw_00','arw_12']

    for datetime_date in date_index: 
        source_date = datetime_date.strftime("%Y%m%d")
        print('\nSource Date',source_date) 
        for m in member:
            source_member = str(m.split('_')[0])
            if 'nam' in source_member:
                source_date_path = source_path+'nam_conusnest/'
            elif source_member in ['arw','nssl','nmmb']:
                source_date_path = source_path+'hiresw_conus{0}/'.format(source_member)

            destination_date_path = destination_path+'{0}/{1}/'.format(m,source_date)
    
            if not os.path.exists(destination_date_path):
                os.makedirs(destination_date_path)

            if '00' in m:	
                if m in ['nam_00']:        
                    for fore_hr in range(0,37):
                        source = source_date_path+'nam_conusnest.{1}00/nam_conusnest.{1}00f{0:02}'.format(
                                                                                            fore_hr,
                                                                                            source_date)
                        destination = destination_date_path+'nam_conusnest_{1}00f0{0:02}.grib2'.format(
                                                                                            fore_hr,
                                                                                            source_date)
                        if os.path.exists(destination):
                            continue 
                        if os.path.exists(source): 
                            os.symlink(source,destination)
                        else:
                            print('Incorrect source file:')
                            print(source)
                elif m in ['arw_00','nssl_00','nmmb_00']:
                    for fore_hr in range(0,37):
                        source = source_date_path+'hiresw_conus{0}.{1}00/hiresw_conus{0}.{1}00f{2:02}'.format(
                                                                                            source_member,
                                                                                            source_date,
                                                                                            fore_hr)
                        destination = destination_date_path+'hiresw_conus{2}_{1}00f0{0:02}.grib2'.format(
                                                                                            fore_hr,
                                                                                            source_date,
                                                                                            source_member)
                
                        if os.path.exists(destination):
                            continue 
                        if os.path.exists(source): 
                            os.symlink(source,destination)
                        else:
                            print('Incorrect source file:')
                            print(source)
            elif '12' in m:
                changed_date = (datetime_date-timedelta(days=1)).strftime("%Y%m%d")
                if 'nam_12' in m:
                    for fore_hr in range(0,37):					
                        source = source_date_path+'nam_conusnest.{1}12/nam_conusnest.{1}12f{0:02}'.format(
                                                                                            (fore_hr+12),
                                                                                            changed_date)
                
                        destination = destination_date_path+'nam_conusnest_{1}12f0{0:02}.grib2'.format(
                                                                                            fore_hr,
                                                                                            source_date)
                    
                        if os.path.exists(destination):
                            continue 
                        if os.path.exists(source): 
                            os.symlink(source,destination)
                        else:
                            print('Incorrect source file:')
                            print(source)
            
                elif m in ['arw_12','nssl_12','nmmb_12']:
                    for fore_hr in range(0,37): 
                        source = source_date_path+'hiresw_conus{0}.{1}12/hiresw_conus{0}.{1}12f{2:02}'.format(
                                                                                            source_member,
                                                                                            changed_date,
                                                                                            (fore_hr+12))

                        destination = destination_date_path+'hiresw_conus{2}_{1}12f0{0:02}.grib2'.format(
                                                                                            fore_hr,
                                                                                            source_date,
                                                                                            source_member)
                        if os.path.exists(destination):
                            continue 
                        if os.path.exists(source): 
                            os.symlink(source,destination)
                        else:
                            print('Incorrect source file:')
                            print(source)
            
    return

if __name__ == "__main__":
        main()
