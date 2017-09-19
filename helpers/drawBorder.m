function border_img = drawBorder(img, start_x, start_y, h, w)

    img(start_y, start_x:start_x + w - 1) = 1;
    img(start_y + h - 1, start_x:start_x + w - 1) = 1;
    img(start_y:start_y + h - 1, start_x) = 1;
    img(start_y:start_y + h - 1, start_x + w - 1) = 1;

    border_img = img;
end