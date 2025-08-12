export function get_inceptionV3_classes_array(str) {
  return str.split(/\s+/).filter((item) => item.length > 0);
}
