function rect = rsz_rt(rect,bd_sz,scale,shift)
r = sqrt(prod(rect(3:4)));
if shift
    rect = round([rect(1)-0.5*scale*r,rect(2)-0.5*scale*r,rect(1)+1*rect(3)+...
        0.5*scale*r,rect(2)+1*rect(4)+0.5*scale*r]);
    x_shift = max([1-rect(1), 0]);
    if x_shift == 0, x_shift = min([bd_sz(2) - rect(3),0]); end
    y_shift = max([1-rect(2), 0]);
    if y_shift == 0, y_shift = min([bd_sz(1) - rect(4),0]); end
    rect([1,3]) = min(max(rect([1,3]) + x_shift,1),bd_sz(2));
    rect([2,4]) = min(max(rect([2,4]) + y_shift,1),bd_sz(1));
else
    rect = round([max([rect(1)-0.5*scale*r,1]),max([rect(2)-0.5*scale*r,1]),...
           min([rect(1)+1*rect(3)+0.5*scale*r,bd_sz(2)]),min([rect(2)+1*rect(4)+0.5*scale*r,bd_sz(1)])]);
end