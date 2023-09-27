#!/usr/bin/env fish

set --global reset (set_color normal)
set --global red (set_color red)
set --global green (set_color green)

function run
    echo $argv | fish_indent --ansi
    eval $argv
end

set --local curl_opts --silent --location
set --local stb_image_h_url https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
if not test -f stb_image.h
    command curl $curl_opts $stb_image_h_url --output stb_image.h
    printf "%sDownloaded%s stb_image.h\n" $green $reset
end

set --local stb_image_write_h_url https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
if not test -f stb_image_write.h
    command curl $curl_opts $stb_image_write_h_url --output stb_image_write.h
    printf "%sDownloaded%s stb_image_write.h\n" $green $reset
end

set --local options (fish_opt --short=d --long=debug)
if not argparse $options -- $argv
    return 1
end
set --local build_type Release
set --query _flag_debug; and set build_type Debug

set --local generator "Unix Makefiles"
command --query ninja; and set generator Ninja
run command cmake -S . -B build -DCMAKE_BUILD_TYPE=$build_type -G $generator


run command cmake --build build --config $build_type
