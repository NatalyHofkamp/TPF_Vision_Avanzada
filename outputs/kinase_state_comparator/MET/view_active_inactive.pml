reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 1220+1221+1222+1223+1224+1225+1226
select dfg_inactive, inactive and resi 1220+1221+1222+1223+1224+1225+1226
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 1268+1269+1270+1271+1272+1273+1274+1275+1276+1277+1278+1279+1280+1281+1282
select alphac_inactive, inactive and resi 1268+1269+1270+1271+1272+1273+1274+1275+1276+1277+1278+1279+1280+1281+1282
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 1222+1223+1224+1225+1226+1227+1228+1229+1230+1231+1232+1233+1236+1237+1238+1239+1244+1245+1246+1247+1248+1249+1250+1251+1252
select activation_loop_inactive, inactive and resi 1222+1223+1224+1225+1226+1227+1228+1229+1230+1231+1232+1233+1234+1235+1236+1237+1238+1239+1240+1241+1242+1243+1244+1245+1246
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select hrd_active, active and resi 1200+1201+1202+1203+1204+1205+1206
select hrd_inactive, inactive and resi 1200+1201+1202+1203+1204+1205+1206
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 1248+1275
select lys_glu_inactive, inactive and resi 1248+1275
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
