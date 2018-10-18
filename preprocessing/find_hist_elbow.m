
figure, plot(spc)

Max = max(spc);
Max_idx = find(spc == Max);
first_pt = [1, spc(1),0];
max_pt = [Max_idx, Max,0];

dists = zeros(1,Max_idx);

for i = 1:Max_idx
    pt = [i, spc(i),0];
    dists(i) = point_to_line(pt, first_pt, max_pt);
end

idx = find(dists == max(dists));

