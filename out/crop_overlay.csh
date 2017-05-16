#!/bin/csh
cd /glade/p/work/ahijevyc/hagelslag/out/ge1px_watertoo
set f=UP_HELI_MAX03_25_UP_HELI_MAX03_25.norpts
foreach d (3km_pbl?_1km_on_3km_pbl?_20????????_$f.png)
	convert -crop 980x900+136+1040 -trim +repage $d overlay.$d
end
montage -geometry 75% -tile 5x overlay*$f.png $f.png
