using System;
using System.Linq;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.IO;
using System.Text.RegularExpressions;

namespace CharDatasetGenerator {
    class Program {
        static void Main(string[] args) {
            const string dirpath_root = "../../../../dataset/";

            FontFamily[] fonts = new InstalledFontCollection().Families;

            string[] skip_lists = new string[] {
                "Bookshelf Symbol 7",
                "HoloLens MDL2 Assets",
                "Marlett",
                "Ink Free",
                "MS Reference Specialty",
                "MT Extra",
                "MV Boli",
                "OpenSymbol",
                "Segoe MDL2 Assets",
                "Segoe Print",
                "Segoe Script",
                "Symbol",
                "Webdings",
                "Wingdings 2",
                "Wingdings 3",
                "Wingdings",
            };

            foreach (FontFamily font in fonts.Where(f => f.IsStyleAvailable(FontStyle.Regular) && f.IsStyleAvailable(FontStyle.Bold))) {
                string fontname = font.Name;

                if (Regex.IsMatch(fontname, @"[^a-zA-Z0-9-_ ]")){
                    continue;
                }
                if (skip_lists.Contains(fontname)) {
                    continue;
                }

                try {
                    foreach (int imagesize in new int[] { 16 }) {
                        /*numeric*/
                        {
                            foreach (bool bold_style in new bool[] { true, false }) {

                                string dirpath = dirpath_root + $"numeric/size_{imagesize}/{fontname.Replace(' ', '-')}/{(bold_style ? "bold" : "regular")}/";

                                Directory.CreateDirectory(dirpath);

                                foreach (int angle in new int[] { -10, -5, 0, +5, +10 }) {
                                    foreach (float aspect in new float[] { 0.8f, 0.9f, 1.0f, 1 / 0.9f, 1 / 0.8f }) {
                                        foreach (float margin in new float[] { 1, 0.5f, 0, -0.5f, -1 }) {
                                            using Bitmap bitmap = DrawString(fontname, "0123456789", imagesize, angle, aspect, margin, bold_style);

                                            bitmap.Save(dirpath + $"img_angle_{angle:p#;m#;z0}_aspect_{(int)(aspect * 100 + 0.5)}_margin_{(margin*10):p#;m#;z0}.png");
                                        }
                                    }
                                }
                            }
                        }

                        /*alphabet*/
                        {
                            foreach (bool bold_style in new bool[] { true, false }) {

                                string dirpath = dirpath_root + $"alphabet/size_{imagesize}/{fontname.Replace(' ', '-')}/{(bold_style ? "bold" : "regular")}/";

                                Directory.CreateDirectory(dirpath);

                                foreach (int angle in new int[] { -10, -5, 0, +5, +10 }) {
                                    foreach (float aspect in new float[] { 0.8f, 0.9f, 1.0f, 1 / 0.9f, 1 / 0.8f }) {
                                        foreach (float margin in new float[] { 1, 0.5f, 0, -0.5f, -1 }) {
                                            using Bitmap bitmap = DrawString(fontname, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-", imagesize, angle, aspect, margin, bold_style);

                                            bitmap.Save(dirpath + $"img_angle_{angle:p#;m#;z0}_aspect_{(int)(aspect * 100 + 0.5)}_margin_{(margin*10):p#;m#;z0}.png");
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Console.WriteLine($"finished {fontname}");
                }
                catch (ArgumentException) {
                    Console.WriteLine($"error! {fontname}");
                }
            }
        }

        static Bitmap DrawChar(string fontname, char c, int imagesize, float angle = 0, float aspect = 1, float margin = 1, bool bold_style = false) {
            string str = $"{c}";

            Bitmap image = new(imagesize, imagesize);

            using GraphicsPath path = new();
            path.AddString(
                str, 
                new FontFamily(fontname), 
                (int)(bold_style ? FontStyle.Bold : FontStyle.Regular), 
                10, 
                Point.Empty, 
                StringFormat.GenericDefault
            );
            
            Matrix matrix_path = new();
            matrix_path.Scale(aspect, 1);
            matrix_path.RotateAt(angle, PointF.Empty);
            path.Transform(matrix_path);

            RectangleF rect = path.GetBounds();
            float size = Math.Max(rect.Width, rect.Height);
            float scale = (imagesize - margin * 2) / size;
            float offset_x = (rect.X + rect.Width / 2) * scale - imagesize / 2;
            float offset_y = (rect.Y + rect.Height / 2) * scale - imagesize / 2;

            using Graphics graphic = Graphics.FromImage(image);

            graphic.Clear(Color.Black);
            graphic.SmoothingMode = SmoothingMode.HighQuality;
            graphic.InterpolationMode = InterpolationMode.High;
            
            Matrix matrix = new();
            matrix.Translate(-offset_x, -offset_y);
            matrix.Scale(scale, scale);
            graphic.Transform = matrix;

            graphic.FillPath(new SolidBrush(Color.White), path);
            
            return image;
        }

        static Bitmap DrawString(string fontname, string str, int imagesize, float angle = 0, float aspect = 1, float margin = 1, bool bold_style = false) {
            Bitmap bitmap_str = new Bitmap(imagesize * str.Length, imagesize);
            using Graphics graphics = Graphics.FromImage(bitmap_str);
            
            int index = 0;
            foreach (char c in str) {
                using Bitmap bitmap = DrawChar(fontname, c, imagesize, angle, aspect, margin, bold_style);

                graphics.DrawImageUnscaled(bitmap, new Point(imagesize * index, 0));
                index++;
            }

            return bitmap_str;
        }
    }
}
