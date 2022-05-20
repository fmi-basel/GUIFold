const pdb_contents = [];
const pdb_names = [];
const chains = [];
const chain_colors = [];
const viewers = [];
{%  for chain, color in chain_color %}
    chains.push("{{chain}}");
    chain_colors.push("{{color}}");
{% endfor %}
{% for pdb in pdb_path_list %}
    pdb_contents.push(`{{pdb[1]}}`);
    pdb_names.push("{{pdb[0]}}");
{% endfor %}

function setupViewer(index) {
      let element = $('#container-' + index);
      let config = { backgroundColor: 'white' };
      let viewer = $3Dmol.createViewer( element, config );
      viewers.push(viewer)
      };

function addModelToViewer(viewer_index, model_index) {
      var m = viewers[viewer_index].addModel();
      m.addMolData(pdb_contents[model_index], 'pdb' );
      viewers[viewer_index].setStyle(
          {
            'cartoon':
            {
                'colorscheme': {
                'prop':'b','gradient': 'roygb','min':50,'max':90
                }
            }
          }
      );


      viewers[viewer_index].zoomTo();                                      /* set camera */
      viewers[viewer_index].render();                                      /* render scene */
}

function setupViewerButtons(viewer_index, element_index) {
      $('#btn_plddt_' + element_index).click(function() {
         viewers[viewer_index].setStyle(
              {
                'cartoon':
                {
                    'colorscheme': {
                    'prop':'b','gradient': 'roygb','min':50,'max':90
                    }
                }
              }
         );
        viewers[viewer_index].render();
      });

      $('#btn_bychains_' + element_index).click(function() {
            console.log("index " + element_index)
            console.log([viewers[viewer_index]])
            let chainsLen = chains.length;
            for (let i = 0; i < chainsLen; i++) {
                     console.log("chains " + chains[i])
                     console.log("chain colors " + chain_colors[i])
                     viewers[viewer_index].setStyle(
                          {
                          'chain': chains[i]
                          },
                          {
                            'cartoon':
                            {
                                'color': chain_colors[i]
                            }
                          }
                     );
            }
            viewers[viewer_index].render();
      });
}

function onTab(evt, blockName, buttonIndex) {
  var i, tabcontent, tablinks;

  tabcontent = document.getElementsByClassName("tabcontent-" + blockName);
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  tablinks = document.getElementsByClassName("tablinks-" + blockName);
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }

  document.getElementById("tabs-" + blockName + "-" + buttonIndex).style.display = "block";
  evt.currentTarget.className += " active";
}

window.addEventListener('load', function () {
    onTab(event, 'pae', '1');
    {% if use_model_viewer %}
    onTab(event, 'model', '1');

    {% for pdb in pdb_path_list %}
        setupViewer({{loop.index}});
        setupViewerButtons({{loop.index-1}}, {{loop.index}});
        addModelToViewer({{loop.index-1}}, {{loop.index-1}});
    {% endfor %}
    let num_viewers = viewers.length;
    let viewer_aligned_num = num_viewers
    setupViewer("aligned");
    setupViewerButtons(viewer_aligned_num, "aligned");
    console.log("num viewers")
    console.log(viewer_aligned_num)
    {% for pdb in pdb_path_list %}
    addModelToViewer(viewer_aligned_num, {{loop.index-1}});
    {% endfor %}

    {% endif %}
    document.getElementById('pae-table-1').style.backgroundColor = "green";
    document.getElementById('plddt-table-1').style.backgroundColor = "green";

    var pae_modal = document.getElementById("pae_modal");


    var pae_link = document.getElementById("pae_examples_link");


    var modal_span = document.getElementsByClassName("modal_close")[0];


    pae_link.onclick = function() {
      pae_modal.style.display = "block";
    }

    modal_span.onclick = function() {
      pae_modal.style.display = "none";
    }

    window.onclick = function(event) {
      if (event.target == pae_modal) {
        pae_modal.style.display = "none";
      }
    }

})