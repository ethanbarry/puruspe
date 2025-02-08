//! This test compares the Rust port's output with the output from
//! the following C program:
//!
//! ```C
//! #include <cerf.h>
//! #include <stdio.h>
//! #include <complex.h>
//!
//! int main() {
//!     double x = 0.0;
//!     for (int i = 0; i < 1000; i++) {
//!         x += 0.1;
//!         double res = erfcx(x);
//!         printf("%.17e,\n", res);
//!     }
//! }
//! ```
//!
//! This can be compiled with the following command:
//! ```bash
//! gcc -o myprogram myprogram.c -lcerf -lm
//! ```

use puruspe::error::erfcx;

const MAX_ERR: f64 = 1e-15;

#[test]
fn test_erfcx() {
    let mut x = 0.;
    for i in 0..1000 {
        x += 0.1;
        let res = erfcx(x);
        let libcerf_val = ERFCX_TABLE[i];

        assert!((libcerf_val - res).abs() < MAX_ERR);
    }
}

const ERFCX_TABLE: [f64; 1000] = [
    8.96456979969126766e-01,
    8.09019519901580830e-01,
    7.34599334567655360e-01,
    6.70787785294761463e-01,
    6.15690344192925787e-01,
    5.67804717386586999e-01,
    5.25930337349440946e-01,
    4.89100589223114945e-01,
    4.56531651323117016e-01,
    4.27583576155806999e-01,
    4.01730460636495146e-01,
    3.78537416929239701e-01,
    3.57642669086090381e-01,
    3.38743540679734612e-01,
    3.21585416454317596e-01,
    3.05952992270940916e-01,
    2.91663297075343386e-01,
    2.78560095636438465e-01,
    2.66509373661672544e-01,
    2.55395676310505804e-01,
    2.45119123345172302e-01,
    2.35592963678613931e-01,
    2.26741562167559141e-01,
    2.18498734537033334e-01,
    2.10806364061143531e-01,
    2.03613247356709132e-01,
    1.96874127331955695e-01,
    1.90548879689991874e-01,
    1.84601825955590743e-01,
    1.79001151181389873e-01,
    1.73718408605408159e-01,
    1.68728096811884276e-01,
    1.64007297572932531e-01,
    1.59535364658930395e-01,
    1.55293655608894215e-01,
    1.51265299832373856e-01,
    1.47434997537184975e-01,
    1.43788844894074508e-01,
    1.40314181600689675e-01,
    1.36999457625061327e-01,
    1.33834116418651949e-01,
    1.30808492311142027e-01,
    1.27913720149762850e-01,
    1.25141655538144908e-01,
    1.22484804273841424e-01,
    1.19936259788385610e-01,
    1.17489647565830274e-01,
    1.15139075660803100e-01,
    1.12879090559758796e-01,
    1.10704637733068642e-01,
    1.08611026313932837e-01,
    1.06593897415364361e-01,
    1.04649195660773339e-01,
    1.02773143558704649e-01,
    1.00962218399499162e-01,
    9.92131313922520464e-02,
    9.75228087954397305e-02,
    9.58883748248142154e-02,
    9.43071361483271142e-02,
    9.27765678005384453e-02,
    9.12943003686838978e-02,
    8.98581083197460134e-02,
    8.84658993528522758e-02,
    8.71157046744158586e-02,
    8.58056701048946907e-02,
    8.45340479360992270e-02,
    8.32991894668105165e-02,
    8.20995381522437034e-02,
    8.09336233097442403e-02,
    7.98000543291530473e-02,
    7.86975153416268752e-02,
    7.76247603054379953e-02,
    7.65806084714790170e-02,
    7.55639401949313727e-02,
    7.45736930628767836e-02,
    7.36088583105870692e-02,
    7.26684775018671403e-02,
    7.17516394511808486e-02,
    7.08574773673972536e-02,
    6.99851662008810493e-02,
    6.91339201773432527e-02,
    6.83029905033862156e-02,
    6.74916632300420899e-02,
    6.66992572618321095e-02,
    6.59251224999804486e-02,
    6.51686381094133543e-02,
    6.44292109000763197e-02,
    6.37062738139153456e-02,
    6.29992845096051662e-02,
    6.23077240377747851e-02,
    6.16310956000855614e-02,
    6.09689233860656876e-02,
    6.03207514821043345e-02,
    5.96861428474616540e-02,
    5.90646783525640012e-02,
    5.84559558752297675e-02,
    5.78595894508138267e-02,
    5.72752084725719549e-02,
    5.67024569388323851e-02,
    5.61409927438226986e-02,
    5.55904870092399847e-02,
    5.50506234538706424e-02,
    5.45210977987673623e-02,
    5.40016172056747706e-02,
    5.34918997465642226e-02,
    5.29916739022936387e-02,
    5.25006780885509117e-02,
    5.20186602073708995e-02,
    5.15453772226368359e-02,
    5.10805947580885503e-02,
    5.06240867164625269e-02,
    5.01756349184838621e-02,
    4.97350287605173649e-02,
    4.93020648897663852e-02,
    4.88765468959823784e-02,
    4.84582850187176076e-02,
    4.80470958692173489e-02,
    4.76428021661074313e-02,
    4.72452324840877680e-02,
    4.68542210148938654e-02,
    4.64696073398351145e-02,
    4.60912362132633452e-02,
    4.57189573563654070e-02,
    4.53526252607116681e-02,
    4.49920990010280256e-02,
    4.46372420566912426e-02,
    4.42879221414787388e-02,
    4.39440110411319224e-02,
    4.36053844583192823e-02,
    4.32719218646097845e-02,
    4.29435063590907218e-02,
    4.26200245332853042e-02,
    4.23013663420461136e-02,
    4.19874249801187663e-02,
    4.16780967640882488e-02,
    4.13732810194366862e-02,
    4.10728799724568716e-02,
    4.07767986467803684e-02,
    4.04849447642923743e-02,
    4.01972286502185680e-02,
    3.99135631421807940e-02,
    3.96338635030298939e-02,
    3.93580473372741949e-02,
    3.90860345109321772e-02,
    3.88177470746473158e-02,
    3.85531091899113734e-02,
    3.82920470582509345e-02,
    3.80344888532393663e-02,
    3.77803646552041725e-02,
    3.75296063885058595e-02,
    3.72821477612711658e-02,
    3.70379242074698259e-02,
    3.67968728312291771e-02,
    3.65589323532866939e-02,
    3.63240430594855210e-02,
    3.60921467512227354e-02,
    3.58631866977647470e-02,
    3.56371075903483825e-02,
    3.54138554979902681e-02,
    3.51933778249309270e-02,
    3.49756232696436659e-02,
    3.47605417853415408e-02,
    3.45480845419191274e-02,
    3.43382038892686980e-02,
    3.41308533219133528e-02,
    3.39259874449024321e-02,
    3.37235619409169407e-02,
    3.35235335385353495e-02,
    3.33258599816123949e-02,
    3.31304999997255947e-02,
    3.29374132796464714e-02,
    3.27465604377952316e-02,
    3.25579029936397119e-02,
    3.23714033440010776e-02,
    3.21870247382304506e-02,
    3.20047312542223600e-02,
    3.18244877752321548e-02,
    3.16462599674664224e-02,
    3.14700142584162790e-02,
    3.12957178159052399e-02,
    3.11233385278241806e-02,
    3.09528449825274242e-02,
    3.07842064498648552e-02,
    3.06173928628262031e-02,
    3.04523747997746189e-02,
    3.02891234672475751e-02,
    3.01276106833040791e-02,
    2.99678088613982256e-02,
    2.98096909947595870e-02,
    2.96532306412621606e-02,
    2.94984019087641304e-02,
    2.93451794409013249e-02,
    2.91935384033183178e-02,
    2.90434544703213580e-02,
    2.88949038119382107e-02,
    2.87478630813705685e-02,
    2.86023094028251892e-02,
    2.84582203597104587e-02,
    2.83155739831857482e-02,
    2.81743487410513020e-02,
    2.80345235269668612e-02,
    2.78960776499878257e-02,
    2.77589908244080842e-02,
    2.76232431598990302e-02,
    2.74888151519348464e-02,
    2.73556876724943485e-02,
    2.72238419610301603e-02,
    2.70932596156962385e-02,
    2.69639225848253422e-02,
    2.68358131586479212e-02,
    2.67089139612447825e-02,
    2.65832079427256521e-02,
    2.64586783716263992e-02,
    2.63353088275177852e-02,
    2.62130831938189382e-02,
    2.60919856508089326e-02,
    2.59720006688301877e-02,
    2.58531130016775440e-02,
    2.57353076801671368e-02,
    2.56185700058794048e-02,
    2.55028855450707588e-02,
    2.53882401227486232e-02,
    2.52746198169047920e-02,
    2.51620109529021377e-02,
    2.50504000980100207e-02,
    2.49397740560837397e-02,
    2.48301198623836969e-02,
    2.47214247785299476e-02,
    2.46136762875881036e-02,
    2.45068620892825439e-02,
    2.44009700953331798e-02,
    2.42959884249119955e-02,
    2.41919054002158844e-02,
    2.40887095421522188e-02,
    2.39863895661339421e-02,
    2.38849343779808446e-02,
    2.37843330699239776e-02,
    2.36845749167101625e-02,
    2.35856493718037234e-02,
    2.34875460636825711e-02,
    2.33902547922259577e-02,
    2.32937655251912491e-02,
    2.31980683947771889e-02,
    2.31031536942711696e-02,
    2.30090118747781104e-02,
    2.29156335420286975e-02,
    2.28230094532646810e-02,
    2.27311305141991160e-02,
    2.26399877760494299e-02,
    2.25495724326412843e-02,
    2.24598758175813124e-02,
    2.23708894014967886e-02,
    2.22826047893403753e-02,
    2.21950137177582482e-02,
    2.21081080525197468e-02,
    2.20218797860070176e-02,
    2.19363210347628783e-02,
    2.18514240370954807e-02,
    2.17671811507381303e-02,
    2.16835848505628273e-02,
    2.16006277263461263e-02,
    2.15183024805858936e-02,
    2.14366019263675968e-02,
    2.13555189852788624e-02,
    2.12750466853710227e-02,
    2.11951781591663939e-02,
    2.11159066417101367e-02,
    2.10372254686654880e-02,
    2.09591280744513241e-02,
    2.08816079904208529e-02,
    2.08046588430804806e-02,
    2.07282743523477635e-02,
    2.06524483298474656e-02,
    2.05771746772447847e-02,
    2.05024473846147082e-02,
    2.04282605288467374e-02,
    2.03546082720839454e-02,
    2.02814848601956151e-02,
    2.02088846212825253e-02,
    2.01368019642141870e-02,
    2.00652313771971166e-02,
    1.99941674263734853e-02,
    1.99236047544492992e-02,
    1.98535380793514807e-02,
    1.97839621929130800e-02,
    1.97148719595859334e-02,
    1.96462623151801748e-02,
    1.95781282656298239e-02,
    1.95104648857839548e-02,
    1.94432673182227504e-02,
    1.93765307720978330e-02,
    1.93102505219964020e-02,
    1.92444219068284607e-02,
    1.91790403287366920e-02,
    1.91141012520284412e-02,
    1.90496002021292056e-02,
    1.89855327645572329e-02,
    1.89218945839186581e-02,
    1.88586813629227287e-02,
    1.87958888614166535e-02,
    1.87335128954396121e-02,
    1.86715493362954871e-02,
    1.86099941096438959e-02,
    1.85488431946090743e-02,
    1.84880926229062485e-02,
    1.84277384779850638e-02,
    1.83677768941896725e-02,
    1.83082040559351643e-02,
    1.82490161968998921e-02,
    1.81902095992333805e-02,
    1.81317805927794784e-02,
    1.80737255543143953e-02,
    1.80160409067992473e-02,
    1.79587231186469111e-02,
    1.79017687030027239e-02,
    1.78451742170388263e-02,
    1.77889362612618160e-02,
    1.77330514788334032e-02,
    1.76775165549038285e-02,
    1.76223282159576854e-02,
    1.75674832291719887e-02,
    1.75129784017861272e-02,
    1.74588105804834755e-02,
    1.74049766507844532e-02,
    1.73514735364507233e-02,
    1.72982981989003234e-02,
    1.72454476366335034e-02,
    1.71929188846690235e-02,
    1.71407090139907084e-02,
    1.70888151310040204e-02,
    1.70372343770024623e-02,
    1.69859639276435728e-02,
    1.69350009924343625e-02,
    1.68843428142259576e-02,
    1.68339866687172296e-02,
    1.67839298639673312e-02,
    1.67341697399167981e-02,
    1.66847036679172167e-02,
    1.66355290502691044e-02,
    1.65866433197680019e-02,
    1.65380439392584545e-02,
    1.64897284011958391e-02,
    1.64416942272158327e-02,
    1.63939389677113419e-02,
    1.63464602014167931e-02,
    1.62992555349995953e-02,
    1.62523226026586552e-02,
    1.62056590657298008e-02,
    1.61592626122979387e-02,
    1.61131309568158729e-02,
    1.60672618397296185e-02,
    1.60216530271100645e-02,
    1.59763023102908984e-02,
    1.59312075055126241e-02,
    1.58863664535726264e-02,
    1.58417770194810743e-02,
    1.57974370921225987e-02,
    1.57533445839236480e-02,
    1.57094974305253578e-02,
    1.56658935904618948e-02,
    1.56225310448440749e-02,
    1.55794077970482763e-02,
    1.55365218724104436e-02,
    1.54938713179251231e-02,
    1.54514542019494414e-02,
    1.54092686139119285e-02,
    1.53673126640260695e-02,
    1.53255844830085444e-02,
    1.52840822218020030e-02,
    1.52428040513023534e-02,
    1.52017481620904289e-02,
    1.51609127641679879e-02,
    1.51202960866979388e-02,
    1.50798963777487201e-02,
    1.50397119040427789e-02,
    1.49997409507090281e-02,
    1.49599818210392442e-02,
    1.49204328362483353e-02,
    1.48810923352383755e-02,
    1.48419586743663567e-02,
    1.48030302272156074e-02,
    1.47643053843707732e-02,
    1.47257825531963375e-02,
    1.46874601576185530e-02,
    1.46493366379108134e-02,
    1.46114104504823251e-02,
    1.45736800676700418e-02,
    1.45361439775338278e-02,
    1.44988006836547743e-02,
    1.44616487049365854e-02,
    1.44246865754100312e-02,
    1.43879128440403820e-02,
    1.43513260745377624e-02,
    1.43149248451704021e-02,
    1.42787077485807094e-02,
    1.42426733916041284e-02,
    1.42068203950907294e-02,
    1.41711473937294694e-02,
    1.41356530358751065e-02,
    1.41003359833777094e-02,
    1.40651949114146742e-02,
    1.40302285083253007e-02,
    1.39954354754477754e-02,
    1.39608145269585933e-02,
    1.39263643897143564e-02,
    1.38920838030959018e-02,
    1.38579715188547076e-02,
    1.38240263009615817e-02,
    1.37902469254575361e-02,
    1.37566321803068645e-02,
    1.37231808652523365e-02,
    1.36898917916725149e-02,
    1.36567637824411420e-02,
    1.36237956717885304e-02,
    1.35909863051650020e-02,
    1.35583345391062492e-02,
    1.35258392411006529e-02,
    1.34934992894584927e-02,
    1.34613135731830279e-02,
    1.34292809918434251e-02,
    1.33974004554494753e-02,
    1.33656708843280948e-02,
    1.33340912090016018e-02,
    1.33026603700676545e-02,
    1.32713773180809479e-02,
    1.32402410134365173e-02,
    1.32092504262547143e-02,
    1.31784045362677744e-02,
    1.31477023327079799e-02,
    1.31171428141973606e-02,
    1.30867249886389581e-02,
    1.30564478731095644e-02,
    1.30263104937539641e-02,
    1.29963118856806399e-02,
    1.29664510928588928e-02,
    1.29367271680173902e-02,
    1.29071391725441124e-02,
    1.28776861763876451e-02,
    1.28483672579598387e-02,
    1.28191815040397724e-02,
    1.27901280096790505e-02,
    1.27612058781083528e-02,
    1.27324142206452686e-02,
    1.27037521566033661e-02,
    1.26752188132024796e-02,
    1.26468133254802155e-02,
    1.26185348362046392e-02,
    1.25903824957881143e-02,
    1.25623554622023154e-02,
    1.25344529008943641e-02,
    1.25066739847040573e-02,
    1.24790178937822203e-02,
    1.24514838155101414e-02,
    1.24240709444200376e-02,
    1.23967784821165973e-02,
    1.23696056371995385e-02,
    1.23425516251871974e-02,
    1.23156156684410892e-02,
    1.22887969960914865e-02,
    1.22620948439639416e-02,
    1.22355084545067733e-02,
    1.22090370767194943e-02,
    1.21826799660821605e-02,
    1.21564363844856391e-02,
    1.21303056001627694e-02,
    1.21042868876204083e-02,
    1.20783795275723558e-02,
    1.20525828068731196e-02,
    1.20268960184525490e-02,
    1.20013184612512945e-02,
    1.19758494401570777e-02,
    1.19504882659417725e-02,
    1.19252342551993077e-02,
    1.19000867302843129e-02,
    1.18750450192515802e-02,
    1.18501084557962513e-02,
    1.18252763791947806e-02,
    1.18005481342466356e-02,
    1.17759230712167068e-02,
    1.17514005457784645e-02,
    1.17269799189577990e-02,
    1.17026605570775769e-02,
    1.16784418317028727e-02,
    1.16543231195868850e-02,
    1.16303038026175226e-02,
    1.16063832677646400e-02,
    1.15825609070279289e-02,
    1.15588361173854515e-02,
    1.15352083007427952e-02,
    1.15116768638828630e-02,
    1.14882412184162570e-02,
    1.14649007807322862e-02,
    1.14416549719505607e-02,
    1.14185032178731674e-02,
    1.13954449489374349e-02,
    1.13724796001692628e-02,
    1.13496066111370142e-02,
    1.13268254259059689e-02,
    1.13041354929933127e-02,
    1.12815362653236745e-02,
    1.12590272001851874e-02,
    1.12366077591860933e-02,
    1.12142774082118341e-02,
    1.11920356173826792e-02,
    1.11698818610118412e-02,
    1.11478156175640990e-02,
    1.11258363696148960e-02,
    1.11039436038099272e-02,
    1.10821368108252136e-02,
    1.10604154853276224e-02,
    1.10387791259358679e-02,
    1.10172272351819617e-02,
    1.09957593194731258e-02,
    1.09743748890541188e-02,
    1.09530734579700376e-02,
    1.09318545440295223e-02,
    1.09107176687684125e-02,
    1.08896623574137961e-02,
    1.08686881388485000e-02,
    1.08477945455759707e-02,
    1.08269811136855673e-02,
    1.08062473828182543e-02,
    1.07855928961326751e-02,
    1.07650172002716430e-02,
    1.07445198453289710e-02,
    1.07241003848167257e-02,
    1.07037583756328103e-02,
    1.06834933780289559e-02,
    1.06633049555790345e-02,
    1.06431926751477720e-02,
    1.06231561068597883e-02,
    1.06031948240689761e-02,
    1.05833084033282628e-02,
    1.05634964243596832e-02,
    1.05437584700247677e-02,
    1.05240941262953244e-02,
    1.05045029822244881e-02,
    1.04849846299181135e-02,
    1.04655386645064862e-02,
    1.04461646841163564e-02,
    1.04268622898432747e-02,
    1.04076310857242078e-02,
    1.03884706787105187e-02,
    1.03693806786411720e-02,
    1.03503606982162984e-02,
    1.03314103529709971e-02,
    1.03125292612494604e-02,
    1.02937170441793676e-02,
    1.02749733256465546e-02,
    1.02562977322699731e-02,
    1.02376898933769032e-02,
    1.02191494409784624e-02,
    1.02006760097453335e-02,
    1.01822692369838222e-02,
    1.01639287626121063e-02,
    1.01456542291367881e-02,
    1.01274452816296680e-02,
    1.01093015677047923e-02,
    1.00912227374957315e-02,
    1.00732084436331061e-02,
    1.00552583412223551e-02,
    1.00373720878217283e-02,
    1.00195493434205411e-02,
    1.00017897704176243e-02,
    9.98409303360004059e-03,
    9.96645880012198872e-03,
    9.94888673948396071e-03,
    9.93137652351209869e-03,
    9.91392782633778932e-03,
    9.89654032437744732e-03,
    9.87921369631253148e-03,
    9.86194762306976704e-03,
    9.84474178780156590e-03,
    9.82759587586665845e-03,
    9.81050957481092480e-03,
    9.79348257434843258e-03,
    9.77651456634265163e-03,
    9.75960524478789414e-03,
    9.74275430579091793e-03,
    9.72596144755273097e-03,
    9.70922637035060050e-03,
    9.69254877652021320e-03,
    9.67592837043804860e-03,
    9.65936485850391220e-03,
    9.64285794912366802e-03,
    9.62640735269213073e-03,
    9.61001278157613825e-03,
    9.59367395009780646e-03,
    9.57739057451792615e-03,
    9.56116237301956814e-03,
    9.54498906569182549e-03,
    9.52887037451372737e-03,
    9.51280602333832125e-03,
    9.49679573787690426e-03,
    9.48083924568343672e-03,
    9.46493627613907880e-03,
    9.44908656043691106e-03,
    9.43328983156679728e-03,
    9.41754582430038775e-03,
    9.40185427517629783e-03,
    9.38621492248541307e-03,
    9.37062750625634877e-03,
    9.35509176824104657e-03,
    9.33960745190053759e-03,
    9.32417430239080615e-03,
    9.30879206654883391e-03,
    9.29346049287875682e-03,
    9.27817933153817116e-03,
    9.26294833432456111e-03,
    9.24776725466187367e-03,
    9.23263584758722382e-03,
    9.21755386973771573e-03,
    9.20252107933740814e-03,
    9.18753723618440099e-03,
    9.17260210163805059e-03,
    9.15771543860630088e-03,
    9.14287701153315702e-03,
    9.12808658638624738e-03,
    9.11334393064454178e-03,
    9.09864881328617189e-03,
    9.08400100477636113e-03,
    9.06940027705548796e-03,
    9.05484640352725682e-03,
    9.04033915904696971e-03,
    9.02587831990993468e-03,
    9.01146366383997685e-03,
    8.99709496997803428e-03,
    8.98277201887090482e-03,
    8.96849459246007452e-03,
    8.95426247407064835e-03,
    8.94007544840040060e-03,
    8.92593330150892941e-03,
    8.91183582080689977e-03,
    8.89778279504539228e-03,
    8.88377401430536992e-03,
    8.86980926998722462e-03,
    8.85588835480042094e-03,
    8.84201106275325777e-03,
    8.82817718914269932e-03,
    8.81438653054432419e-03,
    8.80063888480236284e-03,
    8.78693405101981136e-03,
    8.77327182954865462e-03,
    8.75965202198017781e-03,
    8.74607443113538038e-03,
    8.73253886105543504e-03,
    8.71904511699229424e-03,
    8.70559300539933134e-03,
    8.69218233392210667e-03,
    8.67881291138920298e-03,
    8.66548454780313206e-03,
    8.65219705433135748e-03,
    8.63895024329736257e-03,
    8.62574392817183622e-03,
    8.61257792356389870e-03,
    8.59945204521245926e-03,
    8.58636610997760494e-03,
    8.57331993583208309e-03,
    8.56031334185288449e-03,
    8.54734614821286813e-03,
    8.53441817617248133e-03,
    8.52152924807155279e-03,
    8.50867918732115154e-03,
    8.49586781839554300e-03,
    8.48309496682417633e-03,
    8.47036045918378271e-03,
    8.45766412309053792e-03,
    8.44500578719226298e-03,
    8.43238528116073936e-03,
    8.41980243568406575e-03,
    8.40725708245908948e-03,
    8.39474905418390906e-03,
    8.38227818455043391e-03,
    8.36984430823702399e-03,
    8.35744726090118309e-03,
    8.34508687917232678e-03,
    8.33276300064459720e-03,
    8.32047546386977151e-03,
    8.30822410835019461e-03,
    8.29600877453180903e-03,
    8.28382930379722646e-03,
    8.27168553845886341e-03,
    8.25957732175213941e-03,
    8.24750449782873929e-03,
    8.23546691174992067e-03,
    8.22346440947989421e-03,
    8.21149683787925772e-03,
    8.19956404469848570e-03,
    8.18766587857146060e-03,
    8.17580218900910115e-03,
    8.16397282639300635e-03,
    8.15217764196915842e-03,
    8.14041648784171595e-03,
    8.12868921696680360e-03,
    8.11699568314641455e-03,
    8.10533574102232680e-03,
    8.09370924607007505e-03,
    8.08211605459298842e-03,
    8.07055602371628197e-03,
    8.05902901138116912e-03,
    8.04753487633906583e-03,
    8.03607347814581235e-03,
    8.02464467715595375e-03,
    8.01324833451707436e-03,
    8.00188431216418417e-03,
    7.99055247281412959e-03,
    7.97925267996007700e-03,
    7.96798479786603973e-03,
    7.95674869156142751e-03,
    7.94554422683567754e-03,
    7.93437127023290632e-03,
    7.92322968904660314e-03,
    7.91211935131439469e-03,
    7.90104012581282870e-03,
    7.88999188205221150e-03,
    7.87897449027148006e-03,
    7.86798782143313495e-03,
    7.85703174721820094e-03,
    7.84610614002122404e-03,
    7.83521087294533451e-03,
    7.82434581979732716e-03,
    7.81351085508278677e-03,
    7.80270585400127080e-03,
    7.79193069244150851e-03,
    7.78118524697665476e-03,
    7.77046939485958200e-03,
    7.75978301401819993e-03,
    7.74912598305083338e-03,
    7.73849818122161488e-03,
    7.72789948845593790e-03,
    7.71732978533592323e-03,
    7.70678895309595373e-03,
    7.69627687361820658e-03,
    7.68579342942826271e-03,
    7.67533850369071444e-03,
    7.66491198020484345e-03,
    7.65451374340030740e-03,
    7.64414367833287159e-03,
    7.63380167068017455e-03,
    7.62348760673753779e-03,
    7.61320137341378860e-03,
    7.60294285822713314e-03,
    7.59271194930105902e-03,
    7.58250853536026649e-03,
    7.57233250572663958e-03,
    7.56218375031524493e-03,
    7.55206215963035495e-03,
    7.54196762476152908e-03,
    7.53190003737969072e-03,
    7.52185928973326135e-03,
    7.51184527464432168e-03,
    7.50185788550479429e-03,
    7.49189701627266027e-03,
    7.48196256146820959e-03,
    7.47205441617032357e-03,
    7.46217247601277476e-03,
    7.45231663718056922e-03,
    7.44248679640631405e-03,
    7.43268285096660222e-03,
    7.42290469867844249e-03,
    7.41315223789571626e-03,
    7.40342536750564044e-03,
    7.39372398692529371e-03,
    7.38404799609813662e-03,
    7.37439729549057340e-03,
    7.36477178608855275e-03,
    7.35517136939416779e-03,
    7.34559594742230544e-03,
    7.33604542269731229e-03,
    7.32651969824968559e-03,
    7.31701867761280419e-03,
    7.30754226481965079e-03,
    7.29809036439960264e-03,
    7.28866288137521891e-03,
    7.27925972125906089e-03,
    7.26988079005053568e-03,
    7.26052599423275807e-03,
    7.25119524076946360e-03,
    7.24188843710190256e-03,
    7.23260549114579116e-03,
    7.22334631128827842e-03,
    7.21411080638492425e-03,
    7.20489888575672355e-03,
    7.19571045918712075e-03,
    7.18654543691907982e-03,
    7.17740372965215961e-03,
    7.16828524853960897e-03,
    7.15918990518550017e-03,
    7.15011761164186140e-03,
    7.14106828040585347e-03,
    7.13204182441694399e-03,
    7.12303815705413266e-03,
    7.11405719213317311e-03,
    7.10509884390381468e-03,
    7.09616302704709542e-03,
    7.08724965667261678e-03,
    7.07835864831586180e-03,
    7.06948991793552792e-03,
    7.06064338191088152e-03,
    7.05181895703912640e-03,
    7.04301656053280337e-03,
    7.03423611001719688e-03,
    7.02547752352775980e-03,
    7.01674071950757986e-03,
    7.00802561680483310e-03,
    6.99933213467028234e-03,
    6.99066019275478186e-03,
    6.98200971110679840e-03,
    6.97338061016995921e-03,
    6.96477281078061294e-03,
    6.95618623416540717e-03,
    6.94762080193889013e-03,
    6.93907643610112371e-03,
    6.93055305903531645e-03,
    6.92205059350547731e-03,
    6.91356896265407812e-03,
    6.90510808999973951e-03,
    6.89666789943493667e-03,
    6.88824831522371651e-03,
    6.87984926199942774e-03,
    6.87147066476248309e-03,
    6.86311244887811544e-03,
    6.85477454007417386e-03,
    6.84645686443891444e-03,
    6.83815934841882408e-03,
    6.82988191881645033e-03,
    6.82162450278824689e-03,
    6.81338702784244250e-03,
    6.80516942183691589e-03,
    6.79697161297709764e-03,
    6.78879352981386942e-03,
    6.78063510124149275e-03,
    6.77249625649555421e-03,
    6.76437692515091497e-03,
    6.75627703711967863e-03,
    6.74819652264917972e-03,
    6.74013531231997930e-03,
    6.73209333704387694e-03,
    6.72407052806194006e-03,
    6.71606681694254110e-03,
    6.70808213557941724e-03,
    6.70011641618973707e-03,
    6.69216959131217932e-03,
    6.68424159380503341e-03,
    6.67633235684430941e-03,
    6.66844181392185938e-03,
    6.66056989884351077e-03,
    6.65271654572722332e-03,
    6.64488168900124326e-03,
    6.63706526340228711e-03,
    6.62926720397372018e-03,
    6.62148744606376981e-03,
    6.61372592532372735e-03,
    6.60598257770618046e-03,
    6.59825733946325498e-03,
    6.59055014714485850e-03,
    6.58286093759694827e-03,
    6.57518964795980685e-03,
    6.56753621566632910e-03,
    6.55990057844031606e-03,
    6.55228267429479923e-03,
    6.54468244153035006e-03,
    6.53709981873342241e-03,
    6.52953474477469594e-03,
    6.52198715880743501e-03,
    6.51445700026585979e-03,
    6.50694420886351738e-03,
    6.49944872459169015e-03,
    6.49197048771777991e-03,
    6.48450943878373445e-03,
    6.47706551860446726e-03,
    6.46963866826629277e-03,
    6.46222882912537468e-03,
    6.45483594280618200e-03,
    6.44745995119995474e-03,
    6.44010079646318342e-03,
    6.43275842101609461e-03,
    6.42543276754115912e-03,
    6.41812377898158364e-03,
    6.41083139853985094e-03,
    6.40355556967623241e-03,
    6.39629623610733083e-03,
    6.38905334180463625e-03,
    6.38182683099307660e-03,
    6.37461664814959438e-03,
    6.36742273800171375e-03,
    6.36024504552613976e-03,
    6.35308351594735324e-03,
    6.34593809473621342e-03,
    6.33880872760857368e-03,
    6.33169536052391804e-03,
    6.32459793968398031e-03,
    6.31751641153140316e-03,
    6.31045072274838154e-03,
    6.30340082025532353e-03,
    6.29636665120953450e-03,
    6.28934816300388139e-03,
    6.28234530326549166e-03,
    6.27535801985444619e-03,
    6.26838626086248865e-03,
    6.26142997461174005e-03,
    6.25448910965341678e-03,
    6.24756361476657123e-03,
    6.24065343895682540e-03,
    6.23375853145512280e-03,
    6.22687884171648377e-03,
    6.22001431941877035e-03,
    6.21316491446146767e-03,
    6.20633057696444619e-03,
    6.19951125726677255e-03,
    6.19270690592549702e-03,
    6.18591747371445740e-03,
    6.17914291162309330e-03,
    6.17238317085527007e-03,
    6.16563820282810607e-03,
    6.15890795917080867e-03,
    6.15219239172351811e-03,
    6.14549145253615817e-03,
    6.13880509386729653e-03,
    6.13213326818301280e-03,
    6.12547592815576754e-03,
    6.11883302666328677e-03,
    6.11220451678745180e-03,
    6.10559035181319504e-03,
    6.09899048522739330e-03,
    6.09240487071779305e-03,
    6.08583346217191849e-03,
    6.07927621367598987e-03,
    6.07273307951387235e-03,
    6.06620401416599255e-03,
    6.05968897230829955e-03,
    6.05318790881120838e-03,
    6.04670077873855748e-03,
    6.04022753734658260e-03,
    6.03376814008287510e-03,
    6.02732254258537062e-03,
    6.02089070068132816e-03,
    6.01447257038632398e-03,
    6.00806810790324893e-03,
    6.00167726962130919e-03,
    5.99530001211504189e-03,
    5.98893629214332802e-03,
    5.98258606664841402e-03,
    5.97624929275494907e-03,
    5.96992592776900664e-03,
    5.96361592917714171e-03,
    5.95731925464542279e-03,
    5.95103586201849773e-03,
    5.94476570931864665e-03,
    5.93850875474484854e-03,
    5.93226495667185504e-03,
    5.92603427364926140e-03,
    5.91981666440059666e-03,
    5.91361208782241032e-03,
    5.90742050298336159e-03,
    5.90124186912332604e-03,
    5.89507614565250044e-03,
    5.88892329215051028e-03,
    5.88278326836552939e-03,
    5.87665603421340128e-03,
    5.87054154977676920e-03,
    5.86443977530420792e-03,
    5.85835067120936241e-03,
    5.85227419807009090e-03,
    5.84621031662761748e-03,
    5.84015898778568292e-03,
    5.83412017260970771e-03,
    5.82809383232595589e-03,
    5.82207992832070587e-03,
    5.81607842213942468e-03,
    5.81008927548595094e-03,
    5.80411245022167607e-03,
    5.79814790836474109e-03,
    5.79219561208922477e-03,
    5.78625552372435265e-03,
    5.78032760575369550e-03,
    5.77441182081438274e-03,
    5.76850813169631736e-03,
    5.76261650134139627e-03,
    5.75673689284273305e-03,
    5.75086926944389041e-03,
    5.74501359453811252e-03,
    5.73916983166756264e-03,
    5.73333794452257281e-03,
    5.72751789694088062e-03,
    5.72170965290689452e-03,
    5.71591317655094355e-03,
    5.71012843214853925e-03,
    5.70435538411964618e-03,
    5.69859399702794724e-03,
    5.69284423558012471e-03,
    5.68710606462513330e-03,
    5.68137944915348726e-03,
    5.67566435429654997e-03,
    5.66996074532582616e-03,
    5.66426858765225159e-03,
    5.65858784682550606e-03,
    5.65291848853330912e-03,
    5.64726047860073313e-03,
    5.64161378298951195e-03,
];
