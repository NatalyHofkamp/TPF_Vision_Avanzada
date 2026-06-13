reinitialize
load active_selected.pdb, active
load inactive_selected.pdb, inactive
remove solvent
align inactive, active
hide everything
show cartoon, active or inactive
color green, active
color gray70, inactive
select alphac_active, active and resi 524+525+526+527+528+529+530+531+532+533+534+535+536+537+538
select alphac_inactive, inactive and resi 524+525+526+527+528+529+530+531+532+533+534+535+536+537+538
show sticks, alphac_active or alphac_inactive
color cyan, alphac_active or alphac_inactive
select hrd_active, active and resi 619+620+621+622+623+624+625
select hrd_inactive, inactive and resi 619+620+621+622+623+624+625
show sticks, hrd_active or hrd_inactive
color yellow, hrd_active or hrd_inactive
select lys_glu_active, active and resi 514+531
select lys_glu_inactive, inactive and resi 514+531
show sticks, lys_glu_active or lys_glu_inactive
color red, lys_glu_active or lys_glu_inactive
orient active or inactive
zoom active or inactive
bg_color white
ray 1600, 1200
png active_inactive_pymol.png, dpi=200
