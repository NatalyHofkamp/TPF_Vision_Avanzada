reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 1268+1269+1270+1271+1272+1273+1274
select dfg_inactive, inactive and resi 1268+1269+1270+1271+1272+1273+1274
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 1160+1161+1162+1163+1164+1165+1166+1167+1168+1169+1170+1171+1172+1173+1174
select alphac_inactive, inactive and resi 1160+1161+1162+1163+1164+1165+1166+1167+1168+1169+1170+1171+1172+1173+1174
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 1270+1271+1272+1273+1274+1275+1276+1277+1278+1279+1280+1281+1282+1283+1284+1285+1286+1287+1288+1289+1290+1291+1292+1293+1294
select activation_loop_inactive, inactive and resi 1270+1271+1272+1273+1274+1287+1288+1289+1290+1291+1292+1293+1294+1295+1296+1297+1298+1299+1300+1301+1302+1303+1304+1305+1306
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select hrd_active, active and resi 1245+1246+1247+1248+1249+1250+1251
select hrd_inactive, inactive and resi 1245+1246+1247+1248+1249+1250+1251
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 1150+1167
select lys_glu_inactive, inactive and resi 1150+1167
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
