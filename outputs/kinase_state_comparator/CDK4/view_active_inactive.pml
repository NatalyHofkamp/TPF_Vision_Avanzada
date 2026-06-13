reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 156+157+158+159+160+161+162
select dfg_inactive, inactive and resi 161+162+163+164+165+166+167
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 49+50+51+52+53+54+55+56+57+58+59+60+61+62+63
select alphac_inactive, inactive and resi 54+55+56+57+58+59+60+61+62+63+64+65+66+67+68
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 158+159+160+161+162+163+164+165+166+167+168+169+170+171+173+174+175+176+177+178+179+180+181+182+183
select activation_loop_inactive, inactive and resi 163+164+165+166+167+168+169+170+171+172+173+174+175+176+177+178+179+180+181+182+183+184+185+186+187
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select hrd_active, active and resi 136+137+138+139+140+141+142
select hrd_inactive, inactive and resi 141+142+143+144+145+146+147
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 35+56
select lys_glu_inactive, inactive and resi 40+61
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
