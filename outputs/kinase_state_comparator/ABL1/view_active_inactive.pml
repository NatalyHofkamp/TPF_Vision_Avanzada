reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 379+380+381+382+383+384+385
select dfg_inactive, inactive and resi 379+380+381+382+383+384+385
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 279+280+281+282+283+284+285+286+287+288+289+290+291+292+293
select alphac_inactive, inactive and resi 279+280+281+282+283+284+285+286+287+288+289+290+291+292+293
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 381+382+383+384+385+386+387+388+389+390+391+392+393+394+395+396+397+398+399+400+401+402+403+404+405
select activation_loop_inactive, inactive and resi 381+382+383+384+385+386+387+388+389+390+391+392+393+394+395+396+397+398+399+400+401+402+403+404+405
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select hrd_active, active and resi 359+360+361+362+363+364+365
select hrd_inactive, inactive and resi 359+360+361+362+363+364+365
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 271+286
select lys_glu_inactive, inactive and resi 271+286
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
