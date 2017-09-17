function border_img = drawBorder(test_img, start_x, start_y, h, w)

    test_img(start_y, start_x:start_x + w) = 1;
    test_img(start_y + h, start_x:start_x + w) = 1;
    test_img(start_y:start_y + h, start_x) = 1;
    test_img(start_y:start_y + h, start_x + w) = 1;
    
    border_img = test_img;
end