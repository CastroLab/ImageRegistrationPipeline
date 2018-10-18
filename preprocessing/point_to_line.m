function [ dist ] = point_to_line( pt, v1, v2 )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
a = v1 - v2;
b = pt - v2;
dist = norm(cross(a,b)) / norm(a);
end

