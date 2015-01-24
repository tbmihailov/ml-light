using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MlLight.Models
{
    public class Instance
    {
        public int Id { get; set; }
        public string Category { get; set; }
        public List<string> Tokens { get; set; }
    }
}
