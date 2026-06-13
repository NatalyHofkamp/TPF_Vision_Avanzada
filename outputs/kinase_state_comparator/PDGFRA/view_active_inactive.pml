reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select dfg_active, active and resi 834+835+836+837+838+839+840
select dfg_inactive, inactive and resi 834+835+836+837+838+839+840
show sticks, dfg_active or dfg_inactive
color magenta, dfg_active or dfg_inactive
select alphac_active, active and resi 637+638+639+640+641+642+643+644+645+646+647+648+649+650+651
select alphac_inactive, inactive and resi 637+638+639+640+641+642+643+644+645+646+647+648+649+650+651
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select activation_loop_active, active and resi 836+837+838+839+840+841+842+843+844+845+846+847+848+849+850+851+852+853+854+855+856+857+858+859+860
select activation_loop_inactive, inactive and resi 836+837+838+839+840+841+842+843+844+845+846+847+848+849+850+851+852+853+854+855+856+857+858+859+860
show sticks, activation_loop_active or activation_loop_inactive
color orange, activation_loop_active or activation_loop_inactive
select hrd_active, active and resi 814+815+816+817+818+819+820
select hrd_inactive, inactive and resi 814+815+816+817+818+819+820
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 627+644
select lys_glu_inactive, inactive and resi 627+644
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
